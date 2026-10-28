// Browser checks for the Model Summary screen (run by tests/test_phase_f.py).
// Usage: node phase_f.js <base url> <run id> <screenshot dir>; prints one JSON line.
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
      await page.goto(`${base}/app#run=${runId}/f`);
      await page.waitForSelector("#f-fit svg", { timeout: 60000 });
      if (scheme === "light") {
        out.order = await page.$$eval(".fs > article", els => els.map(e => +e.id.replace("step-", "")));
        out.header = (await page.locator(".runhead .chips").textContent()).replace(/\s+/g, " ").trim();
        out.rail_f = await page.locator(".rail .ph.on").textContent();
        const box = await page.locator("#f-fit svg").boundingBox();
        await page.mouse.move(box.x + box.width * 0.6, box.y + box.height / 2);
        out.fit_tooltip = await page.locator("#f-fit .ftip").textContent();
        out.roi_rows = await page.locator(".froi").first().locator("tbody tr").count();
        out.curves = await page.locator("[data-curve] svg").count();
        out.decays = await page.locator("[data-decay] svg").count();
        out.badges = await page.locator("#step-40 .badge").count();
        out.roi_sum_all = await page.locator("#step-39 .sum").textContent();
        const last52 = page.locator('[data-fperiod="last_52"]');
        if (await last52.count()) {
          await last52.click();
          await page.waitForSelector('[data-fperiod="last_52"][aria-pressed="true"]');
          out.roi_sum_52 = await page.locator("#step-39 .sum").textContent();
        }
        await page.screenshot({ path: `${shots}/phase_f_light.png`, fullPage: true });
      } else {
        await page.screenshot({ path: `${shots}/phase_f_dark.png`, fullPage: true });
      }
      out[`bg_${scheme}`] = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
      await page.close();
    }
    const phone = await browser.newPage({ viewport: { width: 400, height: 860 } });
    await phone.goto(`${base}/app#run=${runId}/f`);
    await phone.waitForSelector("#f-fit svg", { timeout: 60000 });
    out.phone_sideways_scroll = await phone.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
    await phone.screenshot({ path: `${shots}/phase_f_phone.png`, fullPage: true });
  } finally {
    await browser.close();
  }
  console.log(JSON.stringify(out));
})();
