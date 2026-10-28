// Browser checks for the step 24 knots preview (run by tests/test_step24.py).
// Usage: node step24.js <base url> <run id> <screenshot dir>
// Prints one JSON line with what it measured.
const { chromium } = require("playwright");

(async () => {
  const [base, runId, shots] = process.argv.slice(2);
  const out = { errors: [] };
  const browser = await chromium.launch();
  try {
    for (const scheme of ["light", "dark"]) {
      const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, colorScheme: scheme });
      page.on("pageerror", e => out.errors.push(e.message));
      page.on("console", m => m.type() === "error" && out.errors.push(m.text()));
      await page.goto(`${base}/app#run=${runId}/d`);
      await page.waitForSelector("#kp-dyn path", { timeout: 60000 });
      const card = page.locator("#step-24");
      await card.scrollIntoViewIfNeeded();
      if (scheme === "light") {
        // Slider moves: no request may reach the server, and each redraw must be quick.
        const calls = [];
        page.on("request", r => calls.push(r.url()));
        out.redraw_ms = await page.evaluate(() => {
          const s = document.getElementById("kp-range"), times = [];
          for (const k of [1, 6, 19, 27, 39, 12]) {
            const t0 = performance.now();
            s.value = k; s.dispatchEvent(new Event("input"));
            document.getElementById("kp-dyn").getBoundingClientRect();  // force layout
            times.push(performance.now() - t0);
          }
          return Math.max(...times);
        });
        await page.waitForTimeout(300);
        out.server_calls_while_sliding = calls.length;
        await page.evaluate(() => { const s = document.getElementById("kp-range"); s.value = 39; s.dispatchEvent(new Event("input")); });
        out.default_line_label = await page.locator("#kp-dyn text").textContent();
        out.knot_dots_at_39 = await page.locator(".kp-knot").count();
        await page.locator(".kp-knot").nth(3).hover();
        out.tooltip = await page.locator("#kp-tip").textContent();
        out.alert_at_39 = await page.locator("#kp-alerts").textContent();
        out.follow_tile_class = await page.locator(".kp-tile").nth(1).getAttribute("class");
        await page.evaluate(() => { const s = document.getElementById("kp-range"); s.value = 1; s.dispatchEvent(new Event("input")); });
        out.trend_tile = await page.locator(".kp-tile").nth(2).textContent();
        // Apply 19 through the quick button and the Apply button.
        await page.click('#kp-quick button[data-kp="19"]');
        await page.click("#kp-apply");
        await page.waitForFunction(() => document.getElementById("kp") && document.getElementById("kp").dataset.saved === "19", null, { timeout: 60000 });
        await page.waitForSelector("#kp-dyn path");
        out.log = await page.locator(".log").filter({ hasText: "#24" }).textContent();
        out.summary_after_apply = await page.locator("#step-24 .sum").textContent();
        out.pill_after_apply = await page.locator("#step-24 .pill").textContent();
      }
      await card.screenshot({ path: `${shots}/step24_${scheme}.png` });
      out[`bg_${scheme}`] = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
      await page.close();
    }
    const phone = await browser.newPage({ viewport: { width: 400, height: 860 } });
    await phone.goto(`${base}/app#run=${runId}/d`);
    await phone.waitForSelector("#kp-dyn path", { timeout: 60000 });
    await phone.locator("#step-24").scrollIntoViewIfNeeded();
    out.phone_sideways_scroll = await phone.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
    out.phone_chart_width = await phone.evaluate(() => document.querySelector("#kp-chart svg").getBoundingClientRect().width);
    await phone.locator("#step-24").screenshot({ path: `${shots}/step24_phone.png` });
  } finally {
    await browser.close();
  }
  console.log(JSON.stringify(out));
})();
