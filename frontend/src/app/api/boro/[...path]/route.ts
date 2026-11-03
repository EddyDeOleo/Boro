
import { Agent, fetch as undiciFetch } from "undici";
import { NextRequest, NextResponse } from "next/server";

const API_BASE_URL =
  process.env.BORO_API_URL ?? "http://127.0.0.1:8000";

const TIMEOUT_MS = 15 * 60 * 1000;

const apiAgent = new Agent({
  headersTimeout: TIMEOUT_MS,
  bodyTimeout: TIMEOUT_MS,
});

interface RouteContext {
  params: Promise<{ path: string[] }>;
}

async function proxyRequest(
  request: NextRequest,
  context: RouteContext,
  method: "GET" | "POST",
) {
  try {
    const { path } = await context.params;
    const endpoint = path.join("/");

    const targetUrl = new URL(
      `/api/v1/${endpoint}`,
      API_BASE_URL,
    );

    if (method === "GET") {
      targetUrl.search = request.nextUrl.search;
    }

    const headers: Record<string, string> = {
      Accept: "application/json",
      "Cache-Control": "no-store",
    };

    let body: string | ArrayBuffer | undefined;

    if (method === "POST") {
      const contentType =
        request.headers.get("content-type") ?? "";

      if (contentType.includes("multipart/form-data")) {
        headers["Content-Type"] = contentType;
        body = await request.arrayBuffer();
      } else {
        headers["Content-Type"] =
          contentType || "application/json";

        body = await request.text();
      }
    }

    const response = await undiciFetch(targetUrl, {
      method,
      headers,
      body,
      dispatcher: apiAgent,
    });

    const responseBody = await response.text();

    return new NextResponse(responseBody, {
      status: response.status,
      headers: {
        "Content-Type":
          response.headers.get("content-type") ??
          "application/json",
        "Cache-Control": "no-store",
      },
    });
  } catch (error) {
    console.error("Boro API proxy error:", error);

    return NextResponse.json(
      {
        error: "Unable to connect to the Boro API.",
      },
      { status: 502 },
    );
  }
}

export async function POST(
  request: NextRequest,
  context: RouteContext,
) {
  return proxyRequest(request, context, "POST");
}

export async function GET(
  request: NextRequest,
  context: RouteContext,
) {
  return proxyRequest(request, context, "GET");
}
