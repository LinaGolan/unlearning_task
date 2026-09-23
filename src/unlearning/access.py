"""Check Hub authentication separately from model access; never print raw errors."""

import os


def failure_details(exc, phase):
    """Extract only exception class names and HTTP codes, never tokens or URLs."""
    chain, seen = [], set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    types = [type(item).__name__ for item in chain]
    status = next((getattr(getattr(item, "response", None), "status_code", None)
                   for item in chain if getattr(getattr(item, "response", None), "status_code", None)), None)
    if phase == "authentication" and status == 401:
        kind = "invalid_token"
        message = "Hugging Face rejected the token. Replace HF_TOKEN in Colab Secrets with a valid token and rerun the access cell."
    elif "RevisionNotFoundError" in types:
        kind = "revision_not_found"
        message = "The pinned model revision was not found. Keep the current settings and ask for a configuration review."
    elif "GatedRepoError" in types or status == 403:
        kind = "model_access_denied"
        message = (
            "Hugging Face denied access. In the same account that owns the token, check approval for "
            "Llama-3.2-1B-Instruct. If approved, check that the token allows reading this gated model."
        )
    elif status == 401:
        kind = "authentication_or_access_denied"
        message = "The file request was unauthorized. Check both the token and the account's approval for this model."
    elif status == 404:
        kind = "model_or_file_unavailable"
        message = "The requested file is unavailable to this account. Check model access and the pinned model settings."
    elif status == 429:
        kind = "rate_limit"
        message = "Hugging Face is rate-limiting requests. Wait before retrying."
    elif status is not None and status >= 500:
        kind = "hub_service_error"
        message = "Hugging Face returned a server error. Retry later without changing the experiment settings."
    elif any("Connection" in name or "Timeout" in name for name in types):
        kind = "network_error"
        message = "The request could not reach Hugging Face reliably. Check the Colab connection and retry."
    else:
        kind = "unclassified_download_error"
        message = "The download failed for an unclassified reason. Share this sanitized report for diagnosis."
    return {"status": "failed", "phase": phase, "error_kind": kind, "error": message,
            "http_status": status, "exception_types": types}


def diagnose_model_access(settings, token=None):
    """Download one small config file, not weights. Authentication details stay private."""
    report = {"model": settings["id"], "revision": settings["revision"],
              "authentication_verified": False, "is_research_result": False}
    token = (token if token is not None else os.environ.get("HF_TOKEN", "")).strip()
    if not token:
        report.update(status="failed", phase="authentication", error_kind="missing_token",
                      error="No HF_TOKEN was supplied. Add it to Colab Secrets or use the hidden input cell.")
        return report
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        report.update(status="failed", phase="setup", error_kind="missing_dependency",
                      error="huggingface_hub is not installed. Run the notebook installation cell successfully first.")
        return report
    try:
        HfApi().whoami(token=token)
    except Exception as exc:
        report.update(failure_details(exc, "authentication"))
        return report
    report["authentication_verified"] = True
    try:
        hf_hub_download(repo_id=settings["id"], filename="config.json",
                        revision=settings["revision"], token=token, force_download=True)
    except Exception as exc:
        report.update(failure_details(exc, "model_access"))
        return report
    report.update(status="passed", phase="model_access",
                  note="The token is valid and the pinned model configuration is readable. Weights were not downloaded.")
    return report
