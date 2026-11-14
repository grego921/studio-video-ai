import { NextResponse } from "next/server";

// Estado global de geração na memória do servidor
let currentTask = {
  id: null as string | null,
  status: "idle", // idle, processing, completed, error
  progress: 0,
  message: "Pronto para gerar",
  videoUrl: null as string | null,
  filename: null as string | null,
  error: null as string | null,
  startedAt: 0,
};

const sampleVideos = [
  {
    filename: "video_ia_exemplo_reels.mp4",
    url: "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
    downloadUrl: "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
    sizeMb: 14.8,
    createdAt: "14/11/2026 15:33",
  }
];

export async function GET(req: Request) {
  const url = new URL(req.url);
  const action = url.searchParams.get("action");

  if (action === "gallery") {
    return NextResponse.json(sampleVideos);
  }

  return NextResponse.json(currentTask);
}

export async function POST(req: Request) {
  try {
    const formData = await req.formData();
    const scriptText = formData.get("script_text")?.toString() || "";
    const minDuration = Number(formData.get("min_duration") || 5);
    const maxDuration = Number(formData.get("max_duration") || 60);

    const taskId = "task_" + Date.now();
    currentTask = {
      id: taskId,
      status: "processing",
      progress: 5,
      message: `Processando vídeo (${minDuration}s - ${maxDuration}s)...`,
      videoUrl: null,
      filename: `video_gerado_${Date.now()}.mp4`,
      error: null,
      startedAt: Date.now(),
    };

    console.log("Iniciando geração com roteiro:", scriptText.substring(0, 30));

    // Simulação progressiva do pipeline de geração
    return NextResponse.json({
      success: true,
      taskId,
      message: "Geração iniciada com sucesso!",
    });
  } catch (err: any) {
    return NextResponse.json({ success: false, error: err.message }, { status: 500 });
  }
}
