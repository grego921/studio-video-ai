import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json([
    {
      filename: "video_ia_apresentacao_vertical.mp4",
      url: "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
      downloadUrl: "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
      sizeMb: 15.3,
      createdAt: "14/11/2026 15:33:31",
    }
  ]);
}
