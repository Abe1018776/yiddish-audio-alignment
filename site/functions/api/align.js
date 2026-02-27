// Cloudflare Pages Function — proxies alignment requests to RunPod
// Keeps the RunPod API key server-side (set as environment secret)

export async function onRequestPost(context) {
  const { env } = context;

  const RUNPOD_API_KEY = env.RUNPOD_API_KEY;
  const RUNPOD_ENDPOINT_ID = env.RUNPOD_ENDPOINT_ID;

  if (!RUNPOD_API_KEY || !RUNPOD_ENDPOINT_ID) {
    return Response.json(
      { error: "Server misconfigured: missing RunPod credentials" },
      { status: 500 }
    );
  }

  let body;
  try {
    body = await context.request.json();
  } catch {
    return Response.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const { mode, audio_base64, audio_url, audio_format, text, language } = body;

  if (!audio_base64 && !audio_url) {
    return Response.json(
      { error: "Provide 'audio_base64' or 'audio_url'" },
      { status: 400 }
    );
  }

  if (mode === "align" && !text) {
    return Response.json(
      { error: "'text' is required for align mode" },
      { status: 400 }
    );
  }

  // Build RunPod payload
  const input = {
    mode: mode || "transcribe",
    language: language || "yi",
    word_timestamps: true,
  };

  if (audio_base64) {
    input.audio_base64 = audio_base64;
    input.audio_format = audio_format || ".wav";
  } else {
    input.audio_url = audio_url;
  }

  if (mode === "align" && text) {
    input.text = text;
  }

  const runpodUrl = `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync`;

  try {
    // Submit to RunPod (runsync blocks up to ~30s)
    const resp = await fetch(runpodUrl, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${RUNPOD_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ input }),
    });

    const result = await resp.json();

    if (result.status === "COMPLETED") {
      return Response.json(result.output);
    }

    // If still running, poll
    if (result.id && (result.status === "IN_QUEUE" || result.status === "IN_PROGRESS")) {
      const output = await pollJob(result.id, RUNPOD_ENDPOINT_ID, RUNPOD_API_KEY);
      return Response.json(output);
    }

    // Failed
    if (result.status === "FAILED") {
      return Response.json(
        { error: result.error || "RunPod job failed" },
        { status: 502 }
      );
    }

    return Response.json(
      { error: "Unexpected RunPod response", details: result },
      { status: 502 }
    );
  } catch (err) {
    return Response.json(
      { error: `RunPod request failed: ${err.message}` },
      { status: 502 }
    );
  }
}

async function pollJob(jobId, endpointId, apiKey, timeoutMs = 300000) {
  const start = Date.now();
  const statusUrl = `https://api.runpod.ai/v2/${endpointId}/status/${jobId}`;

  while (Date.now() - start < timeoutMs) {
    await new Promise((r) => setTimeout(r, 2000));

    const resp = await fetch(statusUrl, {
      headers: { Authorization: `Bearer ${apiKey}` },
    });
    const result = await resp.json();

    if (result.status === "COMPLETED") return result.output;
    if (result.status === "FAILED") {
      throw new Error(result.error || "Job failed");
    }
  }

  throw new Error(`Job ${jobId} timed out`);
}
