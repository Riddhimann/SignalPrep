export type Health = {
  status: "ready" | "degraded";
  components: Record<string, string>;
  component_status: Record<string, "available" | "degraded" | "unavailable">;
  runtime: {
    provider: string;
    model: string;
    structured_output: string;
    retrieval: string;
    embedding_model: string;
  };
  limitations: string[];
};

export type Scorecard = {
  relevance: number;
  clarity: number;
  structure: number;
  technical_depth: number;
  evidence: number;
};

export type Evidence = {
  chunk_id: string;
  source: "resume" | "job_description" | "rubric";
  text: string;
  score: number;
  retrieval_method: string;
  lexical_score: number | null;
  semantic_score: number | null;
  risk_flags: string[];
};

export type GroundingAudit = {
  status: "grounded" | "partial" | "ungrounded" | "no_evidence";
  citation_valid: boolean;
  support_score: number;
  lexical_support_score: number;
  semantic_support_score: number | null;
  support_method: "lexical" | "semantic_indicator" | "none";
  cited_evidence_ids: string[];
  invalid_evidence_ids: string[];
  risk_flags: string[];
};

export type RuntimeProvenance = {
  generation_provider: string;
  model_name: string;
  structured_output_mode: string;
  retrieval_configured: string;
  retrieval_effective: string;
  embedding_model: string;
  prompt_version: string;
  rubric_version: string;
};

export type CommunicationSignal = {
  label: string;
  confidence: number;
  status: "available" | "low_confidence" | "unavailable";
  explanation: string;
  modality: string;
  probabilities: Record<string, number>;
};

export type Evaluation = {
  scores: Scorecard;
  strengths: string[];
  improvements: string[];
  improved_answer_outline: string[];
  evidence_used: string[];
  next_action_suggestion: string;
  suggested_next_question: string;
  calibration: {
    method: string;
    model_weight: number;
    raw_model_scores: Scorecard;
    observable_scores: Scorecard;
    signals: Record<string, number>;
  } | null;
};

export type Turn = {
  turn_number: number;
  question: string;
  topic: string;
  transcript: string;
  communication_signal: CommunicationSignal;
  evidence: Evidence[];
  question_evidence_ids: string[];
  question_grounding: GroundingAudit | null;
  evaluation: Evaluation;
  controller_action: "probe" | "clarify" | "change_topic" | "finish";
  next_question: string | null;
};

export type Session = {
  session_id: string;
  role: string;
  interview_type: "behavioral" | "technical" | "mixed";
  difficulty: "beginner" | "intermediate" | "advanced";
  max_questions: number;
  required_skills: string[];
  covered_skills: string[];
  current_question: string | null;
  current_topic: string | null;
  current_question_evidence_ids: string[];
  current_question_grounding: GroundingAudit | null;
  turns: Turn[];
  status: string;
  degraded_modes: string[];
  security_events: string[];
  runtime: RuntimeProvenance | null;
};

export type TurnResult = { turn: Turn; session_status: string };

export type FinalReport = {
  session_id: string;
  role: string;
  completed_questions: number;
  score_trends: Record<string, number[]>;
  recurring_strengths: string[];
  recurring_improvements: string[];
  practice_plan: string[];
  disclaimer: string;
  markdown: string;
};
