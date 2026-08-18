"""
FarmHand AI - Hybrid Flock Ledger Anomaly Detection & Clinical Report Engine.
Combines Classical Machine Learning (IsolationForest & Robust Z-Score / MAD),
Deterministic Clinical Percentages & Heuristics, and Local LLM Synthesis (Qwen 2.5 3B).
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest

import database
from database import (
    DB_PATH,
    get_active_farm_memories,
    get_all_health_logs,
    get_current_flock_totals,
    get_db_connection,
    get_farm_by_id,
    get_flock_ledger_history,
    get_system_context_summary,
    save_ledger_anomaly,
)


# -------------------------------------------------------------------
# Feature Extraction & Time-Series Preprocessing
# -------------------------------------------------------------------

def extract_ledger_features(events: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Extracts numerical feature vectors from chronological ledger events for classical ML models.
    Features:
      1. pct_change: ratio of count_change to previous balance
      2. is_loss: binary flag (1 if count_change < 0 else 0)
      3. loss_magnitude: absolute number of lost animals
      4. days_interval: days elapsed since the preceding event
      5. rolling_7d_losses: cumulative losses in the preceding 7 calendar days
    """
    if not events:
        return np.empty((0, 5)), []

    # Sort chronologically ascending
    sorted_events = sorted(events, key=lambda e: e.get("created_at") or "")
    features = []
    processed_metadata = []

    for i, evt in enumerate(sorted_events):
        curr_time = evt.get("created_at") or ""
        try:
            curr_dt = datetime.fromisoformat(curr_time.replace("Z", "+00:00"))
        except Exception:
            curr_dt = datetime.now()

        count_change = int(evt.get("count_change", 0))
        new_total = int(evt.get("new_total", 0))
        prev_total = new_total - count_change
        if prev_total <= 0:
            prev_total = max(1, new_total)

        pct_change = count_change / prev_total
        is_loss = 1.0 if count_change < 0 else 0.0
        loss_magnitude = float(abs(count_change)) if count_change < 0 else 0.0

        # Days interval from previous event
        if i > 0:
            prev_time = sorted_events[i - 1].get("created_at") or ""
            try:
                prev_dt = datetime.fromisoformat(prev_time.replace("Z", "+00:00"))
                days_interval = max(0.01, (curr_dt - prev_dt).total_seconds() / 86400.0)
            except Exception:
                days_interval = 1.0
        else:
            days_interval = 1.0

        # Rolling 7-day losses prior to this event
        seven_days_ago = curr_dt - timedelta(days=7)
        rolling_losses = 0
        for prior_evt in sorted_events[:i]:
            try:
                p_time = prior_evt.get("created_at") or ""
                p_dt = datetime.fromisoformat(p_time.replace("Z", "+00:00"))
                if p_dt >= seven_days_ago and prior_evt.get("count_change", 0) < 0:
                    rolling_losses += abs(int(prior_evt.get("count_change", 0)))
            except Exception:
                continue

        feat_vector = [pct_change, is_loss, loss_magnitude, days_interval, float(rolling_losses)]
        features.append(feat_vector)
        processed_metadata.append({
            "id": evt.get("id"),
            "species": evt.get("species"),
            "event_type": evt.get("event_type"),
            "count_change": count_change,
            "new_total": new_total,
            "created_at": curr_time,
            "notes": evt.get("notes") or ""
        })

    return np.array(features, dtype=float), processed_metadata


# -------------------------------------------------------------------
# Layer 1: Deterministic Rules & Clinical Percentages
# -------------------------------------------------------------------

def evaluate_deterministic_rules(
    events: List[Dict[str, Any]],
    health_logs: List[Dict[str, Any]],
    current_totals: Dict[str, int],
    farm_memories: Optional[List[Dict[str, Any]]] = None
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluates clinical and operational thresholds across the ledger:
      - Single-day mortality > 3% or > 5 poultry / > 1 livestock
      - 7-day rolling mortality > 5% of species herd
      - Multi-day consecutive mortality streaks (>= 2 days)
      - Unexplained count reductions (> 10% drop on count_update/loss)
      - Cross-reference active health symptoms in health_logs and farm_memories
    """
    issues = []
    max_severity = "NORMAL"
    now = datetime.now()
    seven_days_ago = now - timedelta(days=7)

    # 1. Evaluate Recent Ledger Events (Last 7 Days)
    recent_losses_by_species: Dict[str, int] = {}
    mortality_dates_by_species: Dict[str, set] = {}
    unexplained_drops: List[Dict[str, Any]] = []

    for evt in events:
        c_time = evt.get("created_at") or ""
        try:
            evt_dt = datetime.fromisoformat(c_time.replace("Z", "+00:00"))
        except Exception:
            evt_dt = now

        species = (evt.get("species") or "General").capitalize()
        change = int(evt.get("count_change", 0))
        event_type = (evt.get("event_type") or "").lower()
        new_total = int(evt.get("new_total", 0))
        prev_total = new_total - change
        if prev_total <= 0:
            prev_total = max(1, new_total)

        if evt_dt >= seven_days_ago:
            if change < 0:
                abs_loss = abs(change)
                recent_losses_by_species[species] = recent_losses_by_species.get(species, 0) + abs_loss
                mortality_dates_by_species.setdefault(species, set()).add(evt_dt.strftime("%Y-%m-%d"))

                # Check Single-Day Loss Spike
                pct_loss = (abs_loss / prev_total) * 100.0
                is_ruminant = species.lower() in ("goat", "cattle", "sheep", "pig")
                is_critical_spike = pct_loss >= 5.0 or (is_ruminant and abs_loss >= 2) or (not is_ruminant and abs_loss >= 10)
                is_warning_spike = pct_loss >= 3.0 or (is_ruminant and abs_loss >= 1) or (not is_ruminant and abs_loss >= 5)

                date_str = evt_dt.strftime('%Y-%m-%d')
                if is_critical_spike:
                    issues.append({
                        "type": "MORTALITY_SPIKE",
                        "severity": "CRITICAL",
                        "species": species,
                        "description": f"Severe single-event mortality: {abs_loss} {species} lost ({pct_loss:.1f}% drop on {date_str}).",
                        "metrics": {"loss_count": abs_loss, "percentage_drop": round(pct_loss, 1), "date": date_str}
                    })
                    max_severity = "CRITICAL"
                elif is_warning_spike:
                    issues.append({
                        "type": "MORTALITY_ELEVATED",
                        "severity": "WARNING",
                        "species": species,
                        "description": f"Elevated single-event mortality: {abs_loss} {species} lost ({pct_loss:.1f}% drop on {date_str}).",
                        "metrics": {"loss_count": abs_loss, "percentage_drop": round(pct_loss, 1), "date": date_str}
                    })
                    if max_severity != "CRITICAL":
                        max_severity = "WARNING"

            # Check Unexplained Population Drop (count_update or loss with > 10% decrease)
            if event_type in ("count_update", "loss") and change < 0:
                pct_drop = (abs(change) / prev_total) * 100.0
                if pct_drop >= 10.0:
                    unexplained_drops.append({
                        "species": species,
                        "drop_count": abs(change),
                        "percentage": round(pct_drop, 1),
                        "date": evt_dt.strftime("%Y-%m-%d")
                    })

    # 2. Check 7-Day Rolling Cumulative Mortality
    for species, total_loss in recent_losses_by_species.items():
        curr_count = current_totals.get(species, 0)
        baseline = curr_count + total_loss
        if baseline > 0:
            rolling_pct = (total_loss / baseline) * 100.0
            if rolling_pct >= 8.0:
                issues.append({
                    "type": "ROLLING_7D_MORTALITY_CRITICAL",
                    "severity": "CRITICAL",
                    "species": species,
                    "description": f"Critical 7-day cumulative mortality: {total_loss} {species} lost ({rolling_pct:.1f}% of herd over the past week).",
                    "metrics": {"7d_losses": total_loss, "cumulative_percentage": round(rolling_pct, 1)}
                })
                max_severity = "CRITICAL"
            elif rolling_pct >= 4.0:
                issues.append({
                    "type": "ROLLING_7D_MORTALITY_WARNING",
                    "severity": "WARNING",
                    "species": species,
                    "description": f"Elevated 7-day cumulative mortality: {total_loss} {species} lost ({rolling_pct:.1f}% of herd over the past week).",
                    "metrics": {"7d_losses": total_loss, "cumulative_percentage": round(rolling_pct, 1)}
                })
                if max_severity != "CRITICAL":
                    max_severity = "WARNING"

    # 3. Check Consecutive Loss Days
    for species, date_set in mortality_dates_by_species.items():
        if len(date_set) >= 2:
            sorted_dates = sorted([datetime.strptime(d, "%Y-%m-%d") for d in date_set])
            consecutive_streak = 1
            for d_idx in range(1, len(sorted_dates)):
                if (sorted_dates[d_idx] - sorted_dates[d_idx - 1]).days == 1:
                    consecutive_streak += 1
                else:
                    consecutive_streak = 1

            if consecutive_streak >= 2:
                issues.append({
                    "type": "CONSECUTIVE_MORTALITY",
                    "severity": "WARNING" if consecutive_streak == 2 else "CRITICAL",
                    "species": species,
                    "description": f"Multi-day consecutive mortality pattern: Deaths recorded across {consecutive_streak} consecutive days for {species}.",
                    "metrics": {"consecutive_days": consecutive_streak, "distinct_dates": list(date_set)}
                })
                if consecutive_streak >= 3:
                    max_severity = "CRITICAL"
                elif max_severity != "CRITICAL":
                    max_severity = "WARNING"

    # 4. Check Unexplained Drops
    for ud in unexplained_drops:
        issues.append({
            "type": "UNEXPLAINED_POPULATION_DROP",
            "severity": "WARNING",
            "species": ud["species"],
            "description": f"Unexplained inventory reduction: {ud['drop_count']} {ud['species']} ({ud['percentage']}%) drop logged on {ud['date']}.",
            "metrics": ud
        })
        if max_severity != "CRITICAL":
            max_severity = "WARNING"

    # 5. Cross-Reference Health Logs (Symptoms & Illness Context)
    correlated_symptoms: List[Dict[str, Any]] = []
    for hlog in health_logs:
        h_time = hlog.get("timestamp") or ""
        try:
            h_dt = datetime.fromisoformat(h_time.replace("Z", "+00:00"))
        except Exception:
            h_dt = now

        if h_dt >= seven_days_ago:
            notes = (hlog.get("notes") or "").lower()
            event_type = (hlog.get("event_type") or "").lower()
            animal_id = hlog.get("animal_id") or ""
            note_content = hlog.get("notes") or ""

            # Check matching species with mortality
            for species in recent_losses_by_species.keys():
                if species.lower() in notes or species.lower() in animal_id.lower() or species.lower() in event_type:
                    correlated_symptoms.append({
                        "species": species,
                        "event_type": event_type,
                        "notes": note_content,
                        "date": h_dt.strftime("%Y-%m-%d")
                    })
                    issues.append({
                        "type": "HEALTH_SYMPTOM_CORRELATION",
                        "severity": "CRITICAL",
                        "species": species,
                        "description": f"Correlated medical symptom in health logs: '{note_content}' coinciding with recent {species} losses.",
                        "metrics": {"symptom": note_content, "event_type": event_type, "date": h_dt.strftime("%Y-%m-%d")}
                    })
                    max_severity = "CRITICAL"

    # 6. Cross-Reference Persistent Active Farm Memories (Clinical Observations)
    if farm_memories:
        for mem in farm_memories:
            m_time = mem.get("created_at") or ""
            try:
                m_dt = datetime.fromisoformat(m_time.replace("Z", "+00:00"))
            except Exception:
                m_dt = now

            obs = (mem.get("observation") or "").lower()
            m_sp = (mem.get("species") or "").lower()
            obs_content = mem.get("observation") or ""
            cat = mem.get("category", "symptom")

            for species in recent_losses_by_species.keys():
                if species.lower() in m_sp or species.lower() in obs or m_sp in ("general", "unknown"):
                    correlated_symptoms.append({
                        "species": species,
                        "category": cat,
                        "observation": obs_content,
                        "date": m_dt.strftime("%Y-%m-%d")
                    })
                    issues.append({
                        "type": "FARM_MEMORY_SYMPTOM_CORRELATION",
                        "severity": "CRITICAL",
                        "species": species,
                        "description": f"Correlated clinical memory observation: '{obs_content}' coinciding with recent {species} losses.",
                        "metrics": {"observation": obs_content, "category": cat, "date": m_dt.strftime("%Y-%m-%d")}
                    })
                    max_severity = "CRITICAL"

    summary_stats = {
        "recent_losses_by_species": recent_losses_by_species,
        "current_totals": current_totals,
        "correlated_symptoms_count": len(correlated_symptoms),
        "total_issues_count": len(issues)
    }

    return max_severity, issues, summary_stats


# -------------------------------------------------------------------
# Layer 2: Classical Machine Learning Models (Isolation Forest & MAD)
# -------------------------------------------------------------------

def evaluate_classical_ml_anomalies(
    events: List[Dict[str, Any]]
) -> Tuple[bool, float, List[Dict[str, Any]]]:
    """
    Applies Isolation Forest and Robust Z-Score (MAD) across historical features to detect
    multivariate statistical outliers.
    """
    if len(events) < 4:
        return False, 0.0, []

    X, meta = extract_ledger_features(events)
    if X.shape[0] < 4:
        return False, 0.0, []

    # 1. Isolation Forest Outlier Detection
    try:
        clf = IsolationForest(n_estimators=50, contamination=0.15, random_state=42)
        clf.fit(X)
        predictions = clf.predict(X)  # 1 for inlier, -1 for outlier
        decision_scores = clf.decision_function(X)  # Lower score = more anomalous
    except Exception as e:
        print(f"[anomaly_detector] IsolationForest execution error: {e}")
        predictions = np.ones(X.shape[0])
        decision_scores = np.zeros(X.shape[0])

    # 2. Robust Z-Score using Median Absolute Deviation (MAD) on loss magnitudes
    loss_mags = X[:, 2]
    median_loss = np.median(loss_mags)
    mad = np.median(np.abs(loss_mags - median_loss))
    if mad > 0:
        robust_z_scores = 0.6745 * (np.abs(loss_mags - median_loss) / mad)
    else:
        robust_z_scores = np.zeros(len(loss_mags))

    # Evaluate the most recent 3 events
    ml_flagged = []
    latest_score = float(decision_scores[-1]) if len(decision_scores) > 0 else 0.0

    for idx in range(max(0, len(X) - 3), len(X)):
        is_iforest_outlier = bool(predictions[idx] == -1)
        is_zscore_outlier = bool(robust_z_scores[idx] >= 3.0)

        if (is_iforest_outlier or is_zscore_outlier) and X[idx, 1] == 1.0:  # Is a loss event
            m = meta[idx]
            ml_flagged.append({
                "id": m["id"],
                "species": m["species"],
                "loss_count": abs(m["count_change"]),
                "percentage_delta": round(float(X[idx, 0]) * 100, 1),
                "isolation_forest_score": round(float(decision_scores[idx]), 3),
                "robust_z_score": round(float(robust_z_scores[idx]), 2),
                "date": m["created_at"][:10] if m["created_at"] else ""
            })

    is_anomaly = len(ml_flagged) > 0
    return is_anomaly, latest_score, ml_flagged


# -------------------------------------------------------------------
# Layer 3: LLM Clinical Report Synthesis (Qwen 2.5 3B)
# -------------------------------------------------------------------

def synthesize_clinical_report(
    farm_id: str,
    severity: str,
    issues: List[Dict[str, Any]],
    ml_outliers: List[Dict[str, Any]],
    summary_stats: Dict[str, Any],
    language: str = "english"
) -> str:
    """
    Synthesizes the statistical, ML, and heuristic findings into a readable, actionable clinical
    advisory report using the local Qwen 2.5 3B LLM.
    """
    if severity == "NORMAL" and not issues and not ml_outliers:
        return (
            "Flock Health Status: Normal\n\n"
            "All flock and herd count records are consistent with standard operational baselines. "
            "No mortality spikes, consecutive losses, or statistical anomalies were detected across active livestock records."
        )

    from llm_engine import get_llm, _english_logit_bias, N_CTX
    from rag_pipeline import search_knowledge_base

    llm = get_llm()
    farm_context = get_system_context_summary(farm_id)

    # Search knowledge base for veterinary guidance on affected species/symptoms
    affected_species = list(set([i.get("species") for i in issues if i.get("species")] + [m.get("species") for m in ml_outliers if m.get("species")]))
    sp_query = " ".join(affected_species) if affected_species else "livestock"
    rag_hits = search_knowledge_base(f"{sp_query} disease outbreak biosecurity treatment mortality control", top_k=2)
    kb_context = "\n".join([f"- {h.get('text', '')[:350]}" for h in rag_hits if len(h.get('text', '')) > 40])

    # Build structured summary for prompt
    issues_bullets = "\n".join([f"- [{i.get('severity')}] {i.get('description')}" for i in issues])
    if ml_outliers:
        ml_bullets = "\n".join([
            f"- Machine Learning Outlier ({m['species']}): {m['loss_count']} loss ({m['percentage_delta']}%) | Isolation Forest Score: {m['isolation_forest_score']}"
            for m in ml_outliers
        ])
    else:
        ml_bullets = "- No standalone ML statistical outliers."

    system_prompt = (
        "You are FarmHand AI, an expert veterinary epidemiologist and agricultural advisor.\n"
        "Write your advisory report in clear, formal, standard international English.\n"
        "Do NOT use Nigerian Pidgin or slang. Do NOT use emojis.\n"
        "Synthesize the detected flock anomalies into a clean, actionable clinical report with the following 4 sections:\n"
        "1. Situation Overview: State the exact species, mortality numbers, and timeline.\n"
        "2. Statistical & ML Assessment: Summarize percentage losses and anomaly severity.\n"
        "3. Suspected Risks & Causes: Identify potential diseases, biosecurity breaches, or management stress.\n"
        "4. Immediate Actionable Recommendations: Detail practical clinical steps (quarantine, hydration, veterinary consult, sanitation)."
    )

    user_content = (
        f"FARM PROFILE:\n{farm_context}\n\n"
        f"DETECTED FLOCK ANOMALIES:\n{issues_bullets}\n\n"
        f"MACHINE LEARNING ANOMALY ASSESSMENT:\n{ml_bullets}\n\n"
        f"VETERINARY KNOWLEDGE BASE EXCERPTS:\n{kb_context}\n\n"
    )

    prompt = (
        f"<|im_start|>system\n{system_prompt}\n"
        f"<|im_start|>user\n{user_content}\n"
        "<|im_start|>assistant\n# Flock Health Anomaly Advisory\n\n1. Situation Overview:\n"
    )

    if llm is None:
        return (
            f"Flock Anomaly Advisory [{severity}]\n\n"
            f"1. Situation Overview:\n{issues_bullets}\n\n"
            f"2. Statistical & ML Assessment:\n{ml_bullets}\n\n"
            f"3. Suspected Risks & Causes:\nPotential infectious disease outbreak, severe parasitic infestation, or acute environmental stress.\n\n"
            f"4. Immediate Actionable Recommendations:\n- Immediately isolate affected or symptomatic animals.\n- Clean and disinfect all feeding troughs and water containers.\n- Provide oral rehydration and contact a local veterinary officer for diagnostic testing."
        )

    try:
        est_tokens = len(prompt) // 3
        max_gen_tokens = max(100, min(350, N_CTX - est_tokens - 100))

        response = llm.create_completion(
            prompt=prompt,
            max_tokens=max_gen_tokens,
            temperature=0.1,
            repeat_penalty=1.15,
            logit_bias=_english_logit_bias,
            stop=["<|im_end|>", "<|im_start|>"]
        )
        body = response["choices"][0]["text"].strip()
        report = "# Flock Health Anomaly Advisory\n\n1. Situation Overview:\n" + body
        return report
    except Exception as e:
        print(f"[anomaly_detector] LLM synthesis exception: {e}")
        return (
            f"Flock Anomaly Advisory [{severity}]\n\n"
            f"1. Situation Overview:\n{issues_bullets}\n\n"
            f"2. Statistical & ML Assessment:\n{ml_bullets}\n\n"
            f"3. Immediate Actionable Recommendations:\n- Immediately isolate sick animals.\n- Check feed and clean water supplies.\n- Consult a qualified veterinarian immediately."
        )


# -------------------------------------------------------------------
# Main Hybrid Execution Entrypoint
# -------------------------------------------------------------------

def run_flock_anomaly_detection(
    farm_id: str = "default_farm",
    trigger_source: str = "ledger_update",
    language: str = "english",
    db_path: Path = DB_PATH
) -> Dict[str, Any]:
    """
    Main entry point: Runs Layer 1 (Deterministic), Layer 2 (Isolation Forest / MAD ML),
    Layer 3 (LLM Synthesis), and persists the resulting report into `ledger_anomalies`.
    """
    t_start = time.time()
    events = get_flock_ledger_history(farm_id=farm_id, limit=60, db_path=db_path)
    current_totals = get_current_flock_totals(farm_id=farm_id, db_path=db_path)
    health_logs = get_all_health_logs(farm_id=farm_id, db_path=db_path)
    farm_memories = get_active_farm_memories(farm_id=farm_id, limit=30, db_path=db_path)

    # 1. Evaluate Layer 1 Deterministic Rules
    det_severity, det_issues, summary_stats = evaluate_deterministic_rules(
        events=events,
        health_logs=health_logs,
        current_totals=current_totals,
        farm_memories=farm_memories
    )

    # 2. Evaluate Layer 2 Classical ML Models
    ml_anomaly, ml_score, ml_outliers = evaluate_classical_ml_anomalies(events)

    # 3. Aggregate Overall Severity
    if det_severity == "CRITICAL":
        overall_severity = "CRITICAL"
        title = "Critical Flock Health Anomaly Detected"
    elif det_severity == "WARNING" or ml_anomaly:
        overall_severity = "WARNING"
        title = "Flock Mortality & Variance Warning"
    else:
        overall_severity = "NORMAL"
        title = "Flock Status Normal"

    # 4. Layer 3: LLM Clinical Report Synthesis
    report_text = synthesize_clinical_report(
        farm_id=farm_id,
        severity=overall_severity,
        issues=det_issues,
        ml_outliers=ml_outliers,
        summary_stats=summary_stats,
        language=language
    )

    # 5. Persist to Database
    metrics_payload = {
        "overall_severity": overall_severity,
        "trigger_source": trigger_source,
        "deterministic_issues": det_issues,
        "ml_anomaly_detected": ml_anomaly,
        "ml_outliers": ml_outliers,
        "ml_decision_score": round(ml_score, 3),
        "summary_stats": summary_stats,
        "analysis_time_seconds": round(time.time() - t_start, 3)
    }

    record = save_ledger_anomaly(
        farm_id=farm_id,
        severity=overall_severity,
        title=title,
        metrics=metrics_payload,
        report_text=report_text,
        db_path=db_path
    )

    print(f"[anomaly_detector] Farm '{farm_id}' evaluated: {overall_severity} ({len(det_issues)} rules, ML={ml_anomaly}) in {time.time() - t_start:.2f}s")
    return record
