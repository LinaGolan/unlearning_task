import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from unlearning.access import diagnose_model_access, failure_details


class HubError(Exception):
    def __init__(self, status, text="private error detail"):
        super().__init__(text)
        self.response = types.SimpleNamespace(status_code=status)


class GatedRepoError(HubError):
    pass


class RevisionNotFoundError(HubError):
    pass


class AccessTests(unittest.TestCase):
    settings = {"id": "meta-llama/Llama-3.2-1B-Instruct", "revision": "a" * 40}

    def hub(self, login_error=None, download_error=None):
        api = Mock()
        api.whoami.side_effect = login_error
        return types.SimpleNamespace(HfApi=Mock(return_value=api),
                                     hf_hub_download=Mock(side_effect=download_error))

    def test_missing_token_does_not_make_a_request(self):
        hub = self.hub()
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            report = diagnose_model_access(self.settings, "")
        self.assertEqual(report["error_kind"], "missing_token")
        hub.HfApi.assert_not_called()

    def test_invalid_token_does_not_attempt_model_download(self):
        hub = self.hub(login_error=HubError(401, "hf_never_print_this"))
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            report = diagnose_model_access(self.settings, "hf_never_print_this")
        self.assertEqual(report["error_kind"], "invalid_token")
        self.assertNotIn("hf_never_print_this", json.dumps(report))
        hub.hf_hub_download.assert_not_called()

    def test_valid_token_without_model_approval(self):
        hub = self.hub(download_error=GatedRepoError(403))
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            report = diagnose_model_access(self.settings, "test-token")
        self.assertTrue(report["authentication_verified"])
        self.assertEqual(report["error_kind"], "model_access_denied")

    def test_wrapped_http_error_keeps_status_without_private_details(self):
        inner = GatedRepoError(403, "Authorization: Bearer secret")
        outer = OSError("also private")
        outer.__cause__ = inner
        report = failure_details(outer, "tokenizer_download")
        self.assertEqual(report["http_status"], 403)
        self.assertEqual(report["exception_types"], ["OSError", "GatedRepoError"])
        self.assertNotIn("secret", json.dumps(report))

    def test_success_forces_small_pinned_file_access(self):
        hub = self.hub()
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            report = diagnose_model_access(self.settings, "test-token")
        self.assertEqual(report["status"], "passed")
        hub.hf_hub_download.assert_called_once_with(repo_id=self.settings["id"],
            filename="config.json", revision="a" * 40, token="test-token", force_download=True)

    def test_missing_revision_is_not_misreported_as_bad_token(self):
        report = failure_details(RevisionNotFoundError(404), "model_access")
        self.assertEqual(report["error_kind"], "revision_not_found")

    def test_unknown_error_does_not_reveal_its_message(self):
        report = failure_details(ValueError("user@example.com hf_privatevalue"), "model_access")
        self.assertEqual(report["error_kind"], "unclassified_download_error")
        self.assertNotIn("user@example.com", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
