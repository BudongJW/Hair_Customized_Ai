const envBaseUrl =
  typeof process !== "undefined" && process.env
    ? process.env.EXPO_PUBLIC_API_BASE_URL
    : undefined;

export const API_BASE_URL = envBaseUrl || "http://localhost:8080";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
      ...(options.headers || {})
    }
  });

  const text = await response.text();
  const data = parseJson(text);

  if (!response.ok) {
    const message = data?.detail || data?.title || `HTTP ${response.status}`;
    throw new Error(message);
  }

  return data;
}

function parseJson(text) {
  if (!text) {
    return null;
  }

  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

function normalizeUploadUrl(uploadUrl) {
  try {
    const apiOrigin = new URL(API_BASE_URL).origin;
    const url = new URL(uploadUrl);
    if (url.hostname === "localhost" || url.hostname === "127.0.0.1") {
      return `${apiOrigin}${url.pathname}${url.search}`;
    }
  } catch {
    return uploadUrl;
  }
  return uploadUrl;
}

async function uploadLocalImage(upload, imageUri, contentType) {
  const fileResponse = await fetch(imageUri);
  const blob = await fileResponse.blob();
  const response = await fetch(normalizeUploadUrl(upload.uploadUrl), {
    method: "PUT",
    headers: {
      ...(upload.headers || {}),
      "Content-Type": contentType
    },
    body: blob
  });

  if (!response.ok) {
    throw new Error(`이미지 업로드 실패 HTTP ${response.status}`);
  }

  return upload;
}

export const api = {
  health() {
    return request("/api/v1/health");
  },
  loginWithPassword(payload) {
    return request("/api/v1/auth/password/login", {
      method: "POST",
      body: JSON.stringify(payload)
    });
  },
  registerWithPassword(payload) {
    return request("/api/v1/auth/password/register", {
      method: "POST",
      body: JSON.stringify(payload)
    });
  },
  createUser(payload) {
    return request("/api/v1/users", {
      method: "POST",
      body: JSON.stringify(payload)
    });
  },
  createPresignedUrl(payload) {
    return request("/api/v1/uploads/presigned-url", {
      method: "POST",
      body: JSON.stringify(payload)
    });
  },
  createPresignedDownloadUrl(payload) {
    return request("/api/v1/uploads/presigned-download-url", {
      method: "POST",
      body: JSON.stringify(payload)
    });
  },
  uploadLocalImage,
  createFaceProfile(payload) {
    return request("/api/v1/face-profiles", {
      method: "POST",
      body: JSON.stringify(payload)
    });
  },
  listFaceProfiles(userId) {
    return request(`/api/v1/face-profiles?userId=${encodeURIComponent(userId)}`);
  },
  getFaceProfile(profileId) {
    return request(`/api/v1/face-profiles/${profileId}`);
  },
  createFittingJob(payload) {
    return request("/api/v1/fitting-jobs", {
      method: "POST",
      body: JSON.stringify(payload)
    });
  },
  listHairDesigns(userId) {
    return request(`/api/v1/hair-designs?userId=${encodeURIComponent(userId)}`);
  },
  getHairDesign(hairDesignId) {
    return request(`/api/v1/hair-designs/${hairDesignId}`);
  },
  listFittingJobs(userId) {
    return request(`/api/v1/fitting-jobs?userId=${encodeURIComponent(userId)}`);
  },
  getFittingJob(jobId) {
    return request(`/api/v1/fitting-jobs/${jobId}`);
  }
};
