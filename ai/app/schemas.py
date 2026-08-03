from pydantic import BaseModel, Field


class FaceProfileJobRequest(BaseModel):
    profile_id: str = Field(alias="profileId")
    backend_base_url: str | None = Field(default=None, alias="backendBaseUrl")


class HairFittingJobRequest(BaseModel):
    fitting_job_id: str = Field(alias="fittingJobId")
    backend_base_url: str | None = Field(default=None, alias="backendBaseUrl")


class JobResponse(BaseModel):
    accepted: bool
    aggregate_id: str
    status: str
