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
        const tiles = async () => Promise.all([0, 1, 2].map(i => page.locator(".kp-tile").nth(i).getAttribute("class")));
        const setK = k => page.evaluate(k => { const s = document.getElementById("kp-range"); s.value = k; s.dispatchEvent(new Event("input")); }, k);
        out.cards = {};
        for (const k of [1, 12, 24, 39]) {
          await setK(k);
          out.cards[k] = { classes: await tiles(), card2: (await page.locator(".kp-tile").nth(1).textContent()).replace(/\s+/g, " ").trim() };
          if (k === 12 || k === 39) await page.locator(".kp-chart").screenshot({ path: `${shots}/step24_chart_${k}.png` });
        }
        await setK(1);
        out.trend_tile = await page.locator(".kp-tile").nth(2).textContent();
        // Suggest, then preview the suggestion (it may lie beyond the slider).
        await page.click("#kp-suggest");
        await page.waitForSelector("#kp-use-aks", { timeout: 60000 });
        out.suggestion = (await page.locator("#kp-suggest-out").textContent()).replace(/\s+/g, " ").trim();
        await page.click("#kp-use-aks");
        out.suggested_preview = { k: await page.textContent("#kp-k"), apply: await page.textContent("#kp-apply"),
          beyond: await page.isVisible("#kp-beyond"), card2: await page.locator(".kp-tile").nth(1).getAttribute("class") };
        // Apply 19 through the quick button and the Apply button.
        await page.click('#kp-quick button[data-kp="19"]');
        await page.click("#kp-apply");
        await page.waitForFunction(() => document.getElementById("kp") && document.getElementById("kp").dataset.saved === "19", null, { timeout: 60000 });
        await page.waitForSelector("#kp-dyn path");
        out.log = await page.locator(".log").filter({ hasText: "#24" }).textContent();
        out.summary_after_apply = await page.locator("#step-24 > header .sum").textContent();
        out.pill_after_apply = await page.locator("#step-24 > header .pill").textContent();
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
