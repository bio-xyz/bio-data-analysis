export type PlanStep = {
  step_number: number;
  title: string;
  description: string;
  expected_output: string;
};

export type Plan = {
  success: boolean;
  goal: string;
  available_resources: string[];
  steps: PlanStep[];
  expected_artifacts: string[];
  error: string;
};

export type AnswerResponse = {
  summary: string;
  details: string[];
};

export type ArtifactResponse = {
  description: string;
  type: string;
  content: string;
  filename?: string | null;
  path?: string | null;
  id?: string | null;
};

export type TaskResponse = {
  plan: Plan | null;
  answer: AnswerResponse;
  artifacts: ArtifactResponse[];
  success: boolean;
};

export type TaskRequestPayload = {
  task_description: string;
  data_files_description?: string;
};
