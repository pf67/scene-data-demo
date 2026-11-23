type ApiErrorPayload = {
  detail?: string;
  message?: string;
};

export class ApiError extends Error {
  readonly status: number;
  readonly kind: "http" | "network";

  constructor(message: string, status: number, kind: "http" | "network") {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.kind = kind;
  }
}

export function isNetworkError(error: unknown): boolean {
  return error instanceof ApiError && error.kind === "network";
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch {
    throw new ApiError("无法连接场景服务，请确认服务已启动。", 0, "network");
  }
  if (!response.ok) {
    let payload: ApiErrorPayload = {};
    try {
      payload = await response.json() as ApiErrorPayload;
    } catch {
      // A proxy or server may return a non-JSON error page.
    }
    const message = payload.detail || payload.message || `${response.status} ${response.statusText}`;
    throw new ApiError(message, response.status, "http");
  }
  return response.json() as Promise<T>;
}

export function getJson<T>(url: string): Promise<T> {
  return requestJson<T>(url);
}

export function postJson<T>(url: string, body: unknown): Promise<T> {
  return requestJson<T>(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function patchJson<T>(url: string, body: unknown): Promise<T> {
  return requestJson<T>(url, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function putJson<T>(url: string, body: unknown): Promise<T> {
  return requestJson<T>(url, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function deleteJson<T>(url: string): Promise<T> {
  return requestJson<T>(url, { method: "DELETE" });
}

export function postForm<T>(url: string, formData: FormData): Promise<T> {
  return requestJson<T>(url, {
    method: "POST",
    body: formData,
  });
}
