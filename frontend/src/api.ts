import type { FinalReport, Health, Session, TurnResult } from "./types";

const API_BASE = import.meta.env.PROD ? "/api" : "";

async function request<T>(url: string, init?: RequestInit, timeoutMs = 180_000): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${API_BASE}${url}`, { ...init, signal: controller.signal });
    if (!response.ok) {
      let message = `Request failed (${response.status})`;
      try {
        const payload = await response.json();
        message = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail);
      } catch {
        // Keep the HTTP fallback when the response has no JSON body.
      }
      throw new Error(message);
    }
    return response.json() as Promise<T>;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error(
        "The model request exceeded 3 minutes. Check the runtime status and try again.",
      );
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

export const api = {
  health: () => request<Health>("/health", undefined, 10_000),

  createSession: (payload: {
    role: string;
    interview_type: string;
    difficulty: string;
    max_questions: number;
  }) =>
    request<Session>("/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  indexDocuments: (sessionId: string, resume: string | File, jobDescription: string) => {
    if (resume instanceof File) {
      const body = new FormData();
      body.append("resume_file", resume);
      body.append("job_description", jobDescription);
      return request<Session>(`/sessions/${sessionId}/documents`, { method: "POST", body });
    }
    return request<Session>(`/sessions/${sessionId}/documents`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resume_text: resume, job_description: jobDescription }),
    });
  },

  start: (sessionId: string) =>
    request<Session>(`/sessions/${sessionId}/start`, { method: "POST" }),

  answer: (sessionId: string, transcript: string, audio?: File | null) => {
    if (audio) {
      const body = new FormData();
      body.append("audio", audio);
      body.append("transcript", transcript);
      return request<TurnResult>(`/sessions/${sessionId}/answers`, { method: "POST", body });
    }
    return request<TurnResult>(`/sessions/${sessionId}/answers`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transcript }),
    });
  },

  getSession: (sessionId: string) => request<Session>(`/sessions/${sessionId}`),
  complete: (sessionId: string) =>
    request<FinalReport>(`/sessions/${sessionId}/complete`, { method: "POST" }),
};
