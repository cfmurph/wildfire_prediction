import { expect, test } from "@playwright/test";

const PIXEL = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
  "base64",
);

// Basemap and FIRMS WMS tiles are third-party. Stub them so CI does not call those hosts.
// They answer slowly, so tab switches happen while tiles are still streaming, as on a slow connection.
test.beforeEach(async ({ page }) => {
  await page.route(
    /arcgisonline\.com|openstreetmap\.org|firms\.modaps\.eosdis\.nasa\.gov/,
    async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 3000));
      await route.fulfill({ status: 200, contentType: "image/png", body: PIXEL });
    },
  );
});

test("map page switches views against a mocked API", async ({ page }) => {
  const calls: string[] = [];
  const fire = {
    type: "Feature",
    geometry: { type: "Point", coordinates: [-121.5, 50.7] },
    properties: { fire_number: "K12345", fire_status: "Being Held", size_ha: 1200, location: "near Test Lake", fire_cause: "Lightning" },
  };
  await page.route("**/api/v1/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    calls.push(path);
    const features = path.endsWith("/fires/active") ? [fire] : [];
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(path.endsWith("/weather/nearest") ? null : { type: "FeatureCollection", features }),
    });
  });

  await page.goto("/");
  await expect(page.getByText("BC Wildfire Map")).toBeVisible();
  await expect(page.getByRole("button", { name: /Current/ })).toBeVisible();
  await expect(page.getByText("1 active fire · 1,200 ha")).toBeVisible();
  await page.getByRole("button", { name: /K12345/ }).click();
  await expect(page.getByText("near Test Lake", { exact: true })).toBeVisible();
  await expect(page.getByText("No CWFIS station is reporting within 100 km.")).toBeVisible();
  await page.getByRole("button", { name: /All fires/ }).click();
  await expect(page.getByText("1 active fire · 1,200 ha")).toBeVisible();
  await page.getByRole("button", { name: /K12345/ }).click();

  await page.getByRole("button", { name: /History/ }).click();
  await expect(page.getByText("Year")).toBeVisible();
  await expect(page.getByRole("slider")).toHaveValue("2023");
  await expect(page.getByText("2023: 0 fires · 0 ha")).toBeVisible();

  await page.getByRole("button", { name: /Risk/ }).click();
  await expect(page.getByText(/: where fires tend to start/)).toBeVisible();

  // Switching tabs drops the selection, so Current opens on the fire list
  await page.getByRole("button", { name: /Current/ }).click();
  await expect(page.getByRole("slider")).toHaveCount(0);
  await expect(page.getByText("1 active fire · 1,200 ha")).toBeVisible();

  await expect.poll(() => calls.some((path) => path.includes("/api/v1/"))).toBe(true);
});

test("a failed fire feed is reported, not shown as zero fires", async ({ page }) => {
  await page.route("**/api/v1/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    await route.fulfill(
      path.endsWith("/fires/active")
        ? { status: 502, contentType: "application/json", body: '{"detail":"unavailable"}' }
        : { status: 200, contentType: "application/json", body: '{"type":"FeatureCollection","features":[]}' },
    );
  });
  await page.goto("/");
  await expect(page.getByText("Fire data is unavailable right now")).toBeVisible();
  await expect(page.getByText(/active fires? ·/)).toHaveCount(0);
});
