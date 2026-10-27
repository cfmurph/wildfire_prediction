import { expect, test } from "@playwright/test";

const PIXEL = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
  "base64",
);

test("map page switches views against a mocked API", async ({ page }) => {
  const calls: string[] = [];
  // Basemap and FIRMS WMS tiles are third-party. Stub them so CI does not call those hosts.
  await page.route(
    /arcgisonline\.com|openstreetmap\.org|firms\.modaps\.eosdis\.nasa\.gov/,
    async (route) => {
      await route.fulfill({ status: 200, contentType: "image/png", body: PIXEL });
    },
  );
  await page.route("**/api/v1/**", async (route) => {
    calls.push(new URL(route.request().url()).pathname);
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ type: "FeatureCollection", features: [] }),
    });
  });

  await page.goto("/");
  await expect(page.getByText("BC Wildfire Prediction")).toBeVisible();
  await expect(page.getByRole("button", { name: /Current/ })).toBeVisible();
  await expect(page.getByText("Click a fire on the map")).toBeVisible();

  await page.getByRole("button", { name: /History/ }).click();
  await expect(page.getByText("Year")).toBeVisible();
  await expect(page.getByRole("slider")).toHaveValue("2023");

  await page.getByRole("button", { name: /Predictions/ }).click();
  await expect(page.getByRole("slider")).toHaveCount(0);
  await page.getByRole("button", { name: /Risk/ }).click();
  await expect(page.getByText("Click a fire on the map")).toBeVisible();

  await expect.poll(() => calls.some((path) => path.includes("/api/v1/"))).toBe(true);
});
