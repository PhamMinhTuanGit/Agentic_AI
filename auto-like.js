const puppeteer = require("puppeteer");

// Configuration
const CONFIG = {
  url: "https://padlet.com/vonguyetanh1/happy-b-t-y-s-ng-nh-s-ng-kh-e-myxx1sbk0s3ezqi2",
  likeSelector: 'button[data-testid="surfacePostReactionEmojiAccumulatedReactionsButton-2764"]',
  intervalMs: 10000, // 10 seconds
  headless: true, // Set to false to see browser window
  timeout: {
    navigation: 60000,
    selector: 20000,
  },
  retryAttempts: 3,
};

// Utility functions
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const log = {
  info: (message) => console.log(`ℹ️  ${message}`),
  success: (message) => console.log(`✅ ${message}`),
  error: (message) => console.error(`❌ ${message}`),
  warn: (message) => console.warn(`⚠️  ${message}`),
};

// Main auto-like function
async function autoLikePadlet() {
  let browser = null;
  let attempt = 0;

  while (attempt < CONFIG.retryAttempts) {
    try {
      log.info(`Opening browser (Attempt ${attempt + 1}/${CONFIG.retryAttempts})...`);
      
      browser = await puppeteer.launch({
        headless: CONFIG.headless,
        args: [
          '--no-sandbox',
          '--disable-setuid-sandbox',
          '--disable-dev-shm-usage',
        ],
      });

      const page = await browser.newPage();
      
      // Set viewport for consistency
      await page.setViewport({ width: 1280, height: 800 });

      log.info("Navigating to Padlet page...");
      await page.goto(CONFIG.url, {
        waitUntil: "networkidle2",
        timeout: CONFIG.timeout.navigation,
      });

      // Wait for like button to appear
      log.info("Waiting for like button...");
      await page.waitForSelector(CONFIG.likeSelector, {
        timeout: CONFIG.timeout.selector,
      });

      const likeButtons = await page.$$(CONFIG.likeSelector);

      if (likeButtons.length > 0) {
        await likeButtons[0].click();
        log.success(`Like button clicked at ${new Date().toLocaleTimeString()}`);
        return true; // Success
      } else {
        log.warn("No like buttons found!");
        return false;
      }
    } catch (error) {
      attempt++;
      log.error(`Error occurred: ${error.message}`);
      
      if (attempt >= CONFIG.retryAttempts) {
        log.error(`Failed after ${CONFIG.retryAttempts} attempts`);
        return false;
      }
      
      log.info(`Retrying in 5 seconds...`);
      await sleep(5000);
    } finally {
      if (browser) {
        await browser.close();
        log.info("Browser closed");
      }
    }
  }
  
  return false;
}

// Main execution loop
(async () => {
  log.info("🚀 Auto-like Padlet script started");
  log.info(`Interval: ${CONFIG.intervalMs / 1000} seconds`);
  log.info(`Headless mode: ${CONFIG.headless}`);
  
  let iteration = 0;

  while (true) {
    iteration++;
    log.info(`\n━━━ Iteration ${iteration} ━━━`);
    
    const success = await autoLikePadlet();
    
    if (success) {
      log.success("Task completed successfully");
    } else {
      log.warn("Task completed with errors");
    }

    log.info(`Waiting ${CONFIG.intervalMs / 1000} seconds before next run...\n`);
    await sleep(CONFIG.intervalMs);
  }
})().catch((error) => {
  log.error(`Fatal error: ${error.message}`);
  process.exit(1);
});
