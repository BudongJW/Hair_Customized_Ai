from __future__ import annotations

from dataclasses import dataclass

import requests


@dataclass
class BackendClient:
    base_url: str

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    def get_face_profile(self, profile_id: str) -> dict:
        return self._get(f"/api/v1/face-profiles/{profile_id}")

    def update_face_profile(self, profile_id: str, payload: dict) -> dict:
        return self._patch(f"/api/v1/face-profiles/{profile_id}/ai-result", payload)

    def get_fitting_job(self, job_id: str) -> dict:
        return self._get(f"/api/v1/fitting-jobs/{job_id}")

    def get_hair_design(self, hair_design_id: str) -> dict:
        return self._get(f"/api/v1/hair-designs/{hair_design_id}")

    def update_fitting_job(self, job_id: str, payload: dict) -> dict:
        return self._patch(f"/api/v1/fitting-jobs/{job_id}/ai-result", payload)

    def upsert_hair_design(self, payload: dict) -> dict:
        return self._patch("/api/v1/hair-designs/ai-result", payload)

    def _get(self, path: str) -> dict:
        response = requests.get(f"{self.base_url}{path}", timeout=20)
        response.raise_for_status()
        return response.json()

    def _patch(self, path: str, payload: dict) -> dict:
        response = requests.patch(f"{self.base_url}{path}", json=payload, timeout=20)
        response.raise_for_status()
        return response.json()
