import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    status: "idle",
    progress: 0,
    message: "Pronto para gerar",
  });
}
