from __future__ import annotations

from dataclasses import dataclass

import boto3


@dataclass
class S3Storage:
    bucket: str
    region: str

    def __post_init__(self) -> None:
        self._client = boto3.client("s3", region_name=self.region)

    def download_bytes(self, object_key: str) -> bytes:
        response = self._client.get_object(Bucket=self.bucket, Key=object_key)
        return response["Body"].read()

    def upload_bytes(self, object_key: str, data: bytes, content_type: str = "image/jpeg") -> str:
        self._client.put_object(
            Bucket=self.bucket,
            Key=object_key,
            Body=data,
            ContentType=content_type,
        )
        return object_key
