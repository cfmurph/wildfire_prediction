// Serves the static export (out/) for the end-to-end tests, as the API service does in production.
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join } from "node:path";

const root = new URL("../out/", import.meta.url).pathname;
const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".ico": "image/x-icon", ".txt": "text/plain", ".woff2": "font/woff2" };

createServer(async (req, res) => {
  const path = new URL(req.url, "http://localhost").pathname;
  const file = join(root, path.endsWith("/") ? `${path}index.html` : path);
  try {
    const body = await readFile(file);
    res.writeHead(200, { "Content-Type": types[extname(file)] ?? "application/octet-stream" }).end(body);
  } catch {
    res.writeHead(404).end();
  }
}).listen(Number(process.argv[2]), "127.0.0.1");
