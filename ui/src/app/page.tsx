"use client";

import Image from "next/image";
import { useId, useMemo, useState } from "react";
import type { ChangeEvent, FormEvent } from "react";
import {
  Badge,
  Box,
  Button,
  Callout,
  Card,
  Flex,
  Grid,
  Heading,
  Inset,
  Separator,
  Text,
  TextArea,
} from "@radix-ui/themes";
import {
  Cross2Icon,
  DownloadIcon,
  ReloadIcon,
  UploadIcon,
} from "@radix-ui/react-icons";
import type { TaskResponse } from "@/types/task";

type StagedFile = {
  id: string;
  file: File;
};

const IMAGE_EXTENSIONS = new Set(["png", "jpg", "jpeg", "gif", "webp", "svg"]);
const JSON_EXTENSIONS = new Set(["json"]);
const TEXT_EXTENSIONS = new Set(["txt", "csv", "tsv", "md", "log"]);

const MIME_BY_EXTENSION: Record<string, string> = {
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  gif: "image/gif",
  webp: "image/webp",
  svg: "image/svg+xml",
  json: "application/json",
  txt: "text/plain",
  csv: "text/csv",
  tsv: "text/tab-separated-values",
  md: "text/markdown",
  log: "text/plain",
};

type PreviewCategory = "image" | "json" | "text" | "none";

const getFileExtension = (filename?: string | null) => {
  if (!filename) {
    return undefined;
  }
  const parts = filename.split(".");
  if (parts.length < 2) {
    return undefined;
  }
  return parts.pop()?.toLowerCase();
};

const getPreviewCategory = (
  extension: string | undefined,
  type: string
): PreviewCategory => {
  const normalizedType = type.toLowerCase();

  if (extension && IMAGE_EXTENSIONS.has(extension)) {
    return "image";
  }
  if (normalizedType.includes("image") || normalizedType.includes("plot")) {
    return "image";
  }
  if (
    (extension && JSON_EXTENSIONS.has(extension)) ||
    normalizedType.includes("json")
  ) {
    return "json";
  }
  if (
    (extension && TEXT_EXTENSIONS.has(extension)) ||
    normalizedType.includes("text") ||
    normalizedType.includes("table")
  ) {
    return "text";
  }
  return "none";
};

const base64ToUint8Array = (base64: string) => {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i += 1) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
};

const decodeBase64ToString = (base64: string) => {
  try {
    const decoder = new TextDecoder();
    return decoder.decode(base64ToUint8Array(base64));
  } catch (err) {
    console.error("Unable to decode base64 string", err);
    return "";
  }
};

const guessMimeType = (extension: string | undefined, type: string) => {
  if (extension && MIME_BY_EXTENSION[extension]) {
    return MIME_BY_EXTENSION[extension];
  }
  const normalizedType = type.toLowerCase();
  if (normalizedType.includes("image") || normalizedType.includes("plot")) {
    return "image/png";
  }
  if (normalizedType.includes("json")) {
    return "application/json";
  }
  if (normalizedType.includes("text")) {
    return "text/plain";
  }
  return "application/octet-stream";
};

const buildDataUrl = (artifactContent: string, mime: string) => {
  return `data:${mime};base64,${artifactContent}`;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_AGENT_API_BASE ?? "http://localhost:8000/api";

export default function Home() {
  const [taskDescription, setTaskDescription] = useState("");
  const [filesDescription, setFilesDescription] = useState("");
  const [files, setFiles] = useState<StagedFile[]>([]);
  const [response, setResponse] = useState<TaskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const taskDescriptionId = useId();
  const filesInputId = useId();
  const filesDescriptionId = useId();

  const totalUploadSize = useMemo(
    () =>
      files.reduce((acc, current) => {
        return acc + current.file.size;
      }, 0),
    [files]
  );

  const handleFilesChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files?.length) {
      return;
    }

    const newFiles = Array.from(event.target.files).map((file) => ({
      id: crypto.randomUUID(),
      file,
    }));

    setFiles((prev) => [...prev, ...newFiles]);
    event.target.value = "";
  };

  const handleFileRemove = (id: string) => {
    setFiles((prev) => prev.filter((entry) => entry.id !== id));
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!taskDescription.trim()) {
      setError("Task description is required.");
      return;
    }

    const formData = new FormData();
    formData.append("task_description", taskDescription.trim());
    formData.append("data_files_description", filesDescription.trim());
    files.forEach(({ file }) => {
      formData.append("data_files", file);
    });

    setIsSubmitting(true);
    setError(null);
    setResponse(null);

    try {
      const res = await fetch(`${API_BASE_URL}/agent/run`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const message = await res.text();
        throw new Error(
          message || "The agent API returned a non-successful status."
        );
      }

      const payload = (await res.json()) as TaskResponse;
      setResponse(payload);
    } catch (err) {
      const fallback =
        err instanceof Error ? err.message : "Something went wrong.";
      setError(fallback);
    } finally {
      setIsSubmitting(false);
    }
  };

  const formatFileSize = (size: number) => {
    if (size === 0) {
      return "0 B";
    }
    const units = ["B", "KB", "MB", "GB"] as const;
    const index = Math.min(
      Math.floor(Math.log(size) / Math.log(1024)),
      units.length - 1
    );
    return `${(size / 1024 ** index).toFixed(1)} ${units[index]}`;
  };

  return (
    <Box className="min-h-screen bg-slate-950 py-10 text-white">
      <Box className="mx-auto w-full max-w-5xl px-4">
        <Flex direction="column" gap="5">
          <Box>
            <Heading size="8" className="text-white">
              Bio Code Interpreter Demo
            </Heading>
            <Text as="p" color="gray" className="mt-2 max-w-3xl">
              Submit a task, optionally send CSVs or Excel files, and the code
              interpreter + data science agent will plan, execute, and respond
              end-to-end. All fields mirror the FastAPI <code>TaskRequest</code>
              contract.
            </Text>
          </Box>

          <Card size="4" className="bg-slate-900/70 backdrop-blur">
            <Heading size="5" className="text-white">
              Run an Agent Task
            </Heading>
            <Text as="p" color="gray" className="mt-1">
              Task description is required. Files and their joint description
              are optional.
            </Text>

            <form onSubmit={handleSubmit} className="mt-6 space-y-6">
              <Box>
                <Flex align="baseline" justify="between">
                  <Text as="label" htmlFor={taskDescriptionId} weight="bold">
                    Task description
                  </Text>
                  <Badge color="jade">required</Badge>
                </Flex>
                <TextArea
                  id={taskDescriptionId}
                  name="task_description"
                  placeholder="Describe what the agent should do..."
                  value={taskDescription}
                  onChange={(event) => setTaskDescription(event.target.value)}
                  required
                  rows={4}
                  className="mt-2"
                />
              </Box>

              <Box>
                <Flex align="baseline" justify="between">
                  <Text as="label" htmlFor={filesInputId} weight="bold">
                    Data files (optional)
                  </Text>
                  <Text color="gray">Multiple files supported</Text>
                </Flex>

                <label
                  htmlFor={filesInputId}
                  className="mt-2 flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-slate-600 bg-slate-900/60 px-6 py-8 text-center transition hover:border-slate-300"
                >
                  <UploadIcon className="mb-3 h-6 w-6" />
                  <Text weight="medium">Click to choose files</Text>
                  <Text color="gray" size="2">
                    CSV, TSV, Excel, or image files
                  </Text>
                  <input
                    id={filesInputId}
                    name="data_files"
                    type="file"
                    multiple
                    onChange={handleFilesChange}
                    className="hidden"
                  />
                </label>

                {files.length > 0 && (
                  <Box className="mt-4 space-y-3">
                    <Flex justify="between" align="center">
                      <Text weight="medium">
                        Selected files ({files.length})
                      </Text>
                      <Text color="gray" size="2">
                        Total {formatFileSize(totalUploadSize)}
                      </Text>
                    </Flex>
                    <Grid
                      gap="3"
                      columns={{ initial: "1", sm: "2" }}
                      className="w-full"
                    >
                      {files.map(({ id, file }) => (
                        <Card
                          key={id}
                          size="2"
                          className="relative bg-slate-900"
                        >
                          <Flex direction="column" gap="2">
                            <Text weight="medium" className="truncate">
                              {file.name}
                            </Text>
                            <Text size="2" color="gray">
                              {file.type || "unknown"} ·{" "}
                              {formatFileSize(file.size)}
                            </Text>
                          </Flex>
                          <Button
                            type="button"
                            variant="ghost"
                            color="red"
                            className="absolute right-2 top-2"
                            onClick={() => handleFileRemove(id)}
                            highContrast
                          >
                            <Cross2Icon aria-label="Remove file" />
                          </Button>
                        </Card>
                      ))}
                    </Grid>
                  </Box>
                )}
              </Box>

              <Box>
                <Flex align="baseline" justify="between">
                  <Text as="label" htmlFor={filesDescriptionId} weight="bold">
                    Files description (optional)
                  </Text>
                  <Text color="gray">
                    Describe schemas, context, or units for each upload.
                  </Text>
                </Flex>
                <TextArea
                  id={filesDescriptionId}
                  name="data_files_description"
                  placeholder="Example: dose_response.csv has columns conc_uM and viability_percent"
                  value={filesDescription}
                  onChange={(event) => setFilesDescription(event.target.value)}
                  rows={3}
                  className="mt-2"
                />
              </Box>

              <Flex gap="4" align="center">
                <Button type="submit" disabled={isSubmitting} highContrast>
                  {isSubmitting ? (
                    <Flex align="center" gap="2">
                      <ReloadIcon className="h-4 w-4 animate-spin" />
                      <span>Running agent...</span>
                    </Flex>
                  ) : (
                    "Send to agent"
                  )}
                </Button>
                {response && (
                  <Text color="jade" weight="medium">
                    Agent run complete
                  </Text>
                )}
              </Flex>
            </form>
          </Card>

          {error && (
            <Callout.Root color="red">
              <Callout.Icon>
                <Cross2Icon />
              </Callout.Icon>
              <Callout.Text>{error}</Callout.Text>
            </Callout.Root>
          )}

          {response && (
            <Card size="4" className="bg-slate-900/80 backdrop-blur">
              <Heading size="5" className="text-white">
                Agent Response
              </Heading>
              <Text color={response.success ? "jade" : "red"} weight="medium">
                {response.success ? "Success" : "Failure"}
              </Text>

              <Box className="mt-6 space-y-8">
                {response.plan && (
                  <section>
                    <Flex justify="between" align="center">
                      <Heading size="4">Plan</Heading>
                      <Badge color={response.plan.success ? "green" : "red"}>
                        {response.plan.success ? "ready" : "blocked"}
                      </Badge>
                    </Flex>
                    {response.plan.goal && (
                      <Text as="p" className="mt-2" color="gray">
                        Goal: {response.plan.goal}
                      </Text>
                    )}
                    {response.plan.available_resources.length > 0 && (
                      <Box className="mt-3">
                        <Text weight="medium">Available resources</Text>
                        <ul className="mt-1 list-disc space-y-1 pl-5 text-sm text-slate-200">
                          {response.plan.available_resources.map((item) => (
                            <li key={item}>{item}</li>
                          ))}
                        </ul>
                      </Box>
                    )}
                    <Separator my="4" className="bg-slate-800" />
                    <Grid gap="3" columns={{ initial: "1", sm: "2" }}>
                      {response.plan.steps.map((step) => (
                        <Card
                          key={step.step_number}
                          className="bg-slate-950/50"
                        >
                          <Text weight="medium" color="gray">
                            Step {step.step_number}: {step.title}
                          </Text>
                          <Text as="p" className="mt-2 text-sm text-slate-200">
                            {step.description}
                          </Text>
                          {step.expected_output && (
                            <Text
                              as="p"
                              className="mt-2 text-xs text-slate-400"
                            >
                              Expected output: {step.expected_output}
                            </Text>
                          )}
                        </Card>
                      ))}
                    </Grid>
                  </section>
                )}

                <section>
                  <Heading size="4">Answer</Heading>
                  {response.answer?.summary && (
                    <Text as="p" className="mt-2">
                      {response.answer.summary}
                    </Text>
                  )}
                  {response.answer?.details?.length > 0 && (
                    <ul className="mt-3 list-disc space-y-2 pl-5 text-sm text-slate-200">
                      {response.answer.details.map((detail) => (
                        <li key={`${detail}-${detail.length}`}>{detail}</li>
                      ))}
                    </ul>
                  )}
                </section>

                {response.artifacts.length > 0 && (
                  <section>
                    <Heading size="4">Artifacts</Heading>
                    <Grid
                      gap="3"
                      columns={{ initial: "1", sm: "2" }}
                      className="mt-3"
                    >
                      {response.artifacts.map((artifact, index) => {
                        const filename =
                          artifact.filename ?? `artifact-${index + 1}`;
                        const extension = getFileExtension(filename);
                        const previewCategory = getPreviewCategory(
                          extension,
                          artifact.type
                        );
                        const mime = guessMimeType(extension, artifact.type);

                        const handleDownload = () => {
                          if (!artifact.content) {
                            return;
                          }
                          const blob = new Blob(
                            [base64ToUint8Array(artifact.content)],
                            {
                              type: mime,
                            }
                          );
                          const url = URL.createObjectURL(blob);
                          const link = document.createElement("a");
                          link.href = url;
                          link.download = filename;
                          document.body.appendChild(link);
                          link.click();
                          link.remove();
                          URL.revokeObjectURL(url);
                        };

                        const renderPreview = () => {
                          if (!artifact.content) {
                            return null;
                          }
                          if (previewCategory === "image") {
                            return (
                              <Box className="mt-3 overflow-hidden rounded-lg border border-slate-800 bg-slate-950/60">
                                <Image
                                  src={buildDataUrl(artifact.content, mime)}
                                  alt={artifact.description || filename}
                                  width={1024}
                                  height={768}
                                  className="h-auto w-full object-contain"
                                  unoptimized
                                />
                              </Box>
                            );
                          }
                          if (previewCategory === "json") {
                            let pretty = "";
                            try {
                              const parsed = JSON.parse(
                                decodeBase64ToString(artifact.content)
                              );
                              pretty = JSON.stringify(parsed, null, 2);
                            } catch {
                              pretty = decodeBase64ToString(artifact.content);
                            }
                            return (
                              <pre className="mt-3 max-h-80 overflow-auto rounded-lg bg-slate-950/60 p-4 text-xs">
                                {pretty}
                              </pre>
                            );
                          }
                          if (previewCategory === "text") {
                            return (
                              <pre className="mt-3 max-h-80 overflow-auto rounded-lg bg-slate-950/60 p-4 text-xs">
                                {decodeBase64ToString(artifact.content)}
                              </pre>
                            );
                          }
                          return (
                            <Text as="p" size="2" color="gray" className="mt-3">
                              Preview not available for this artifact type.
                            </Text>
                          );
                        };

                        return (
                          <Card
                            key={`${artifact.id ?? artifact.filename ?? index}`}
                            className="bg-slate-950/50"
                          >
                            <Inset>
                              <Flex justify="between" align="start" gap="3">
                                <Box>
                                  <Text weight="medium">
                                    {artifact.description ||
                                      artifact.filename ||
                                      `Artifact ${index + 1}`}
                                  </Text>
                                  <Text size="2" color="gray">
                                    Type: {artifact.type}
                                  </Text>
                                  {artifact.path && (
                                    <Text
                                      as="p"
                                      size="2"
                                      className="mt-1 text-slate-400"
                                    >
                                      Path: {artifact.path}
                                    </Text>
                                  )}
                                  {artifact.filename && (
                                    <Text
                                      as="p"
                                      size="2"
                                      className="text-slate-400"
                                    >
                                      Filename: {artifact.filename}
                                    </Text>
                                  )}
                                </Box>
                                <Button
                                  type="button"
                                  variant="soft"
                                  size="1"
                                  highContrast
                                  disabled={!artifact.content}
                                  onClick={handleDownload}
                                >
                                  <DownloadIcon />
                                </Button>
                              </Flex>
                              {renderPreview()}
                            </Inset>
                          </Card>
                        );
                      })}
                    </Grid>
                  </section>
                )}
              </Box>
            </Card>
          )}
        </Flex>
      </Box>
    </Box>
  );
}
