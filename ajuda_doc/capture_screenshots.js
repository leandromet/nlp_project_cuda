const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

// URLs extracted from your document
const urls = [
    // Indigenous Communities and Organizations
    { name: 'Adams Lake Indian Band', url: 'https://adamslakeband.org' },
    { name: 'Esketemc First Nation', url: 'https://www.esketemc.ca/' },
    { name: 'Alkali Resource Management', url: 'https://www.armltd.org/' },
    { name: 'First Nations Summit', url: 'https://fns.bc.ca/' },
    { name: 'Union of BC Indian Chiefs', url: 'https://www.ubcic.bc.ca/' },
    { name: 'Lower Similkameen Indian Band', url: 'https://www.lsib.net' },
    { name: 'Kitaskinaw Education Authority', url: 'https://kitaskinaw.com/' },
    { name: 'Neskonlith Indian Band', url: 'https://neskonlith.net/' },
    { name: 'Okanagan Indian Band', url: 'https://www.okib.ca' },
    { name: 'Okanagan Nation Alliance', url: 'https://www.syilx.org' },
    { name: 'Okanagan Training and Consulting', url: 'https://okcp.ca' },
    { name: 'Penticton Indian Band', url: 'https://www.pib.ca' },
    { name: 'SD73 Aboriginal Education', url: 'https://dallas.sd73.bc.ca/en/our-schools-programs/aboriginal-education.aspx' },
    { name: 'Splatsin First Nation', url: 'http://www.splatsin.ca/' },
    { name: 'Tsideldel First Nation', url: 'https://www.tsideldel.org/' },
    { name: 'Upper Nicola Band', url: 'https://uppernicola.com' },
    { name: 'Westbank First Nation', url: 'https://www.wfn.ca' },
    
    // Provincial Government
    { name: 'BC Ministry of Forests', url: 'https://www2.gov.bc.ca/gov/content/governments/government-structure/ministries-organizations/ministries/forests' },
    
    // Forestry and Policy Institutions
    { name: 'BC Forestry Innovation Investment', url: 'https://www.bcfii.ca' },
    { name: 'Council of Forest Industries', url: 'https://www.cofi.org' },
    { name: 'First Nations Forestry Council', url: 'https://www.fnforestrycouncil.ca' },
    
    // Academic and Research Partners
    { name: 'Selkirk College ARIC', url: 'https://selkirk.ca/ari' },
    { name: 'Thompson Rivers University', url: 'https://www.tru.ca/nrs/' },
    { name: 'UBC Okanagan Indigenous Initiatives', url: 'https://equity.ok.ubc.ca/indigenous-initiatives/' },
    { name: 'University of Victoria', url: 'https://www.uvic.ca/socialsciences/environmental/' },
    
    // Federal Government
    { name: 'Crown-Indigenous Relations Canada', url: 'https://www.rcaanc-cirnac.gc.ca/' },
    { name: 'Environment and Climate Change Canada', url: 'https://www.canada.ca/en/environment-climate-change.html' },
    { name: 'Indigenous Services Canada', url: 'https://www.sac-isc.gc.ca' },
    { name: 'Natural Resources Canada', url: 'https://natural-resources.canada.ca' },
    
    // NGOs and Environmental Groups
    { name: 'CPAWS BC', url: 'https://cpawsbc.org' },
    { name: 'Ecojustice', url: 'https://ecojustice.ca' },
    { name: 'Yellowstone to Yukon', url: 'https://y2y.net' },
    
    // Corporate Forestry
    { name: 'BC Community Forest Association', url: 'https://bccfa.ca/community-forests-and-value-added-enterprises/' },
    { name: 'Forest Professionals BC', url: 'https://www.fpbc.ca/' },
    { name: 'Mercer', url: 'https://mercerint.com' },
    { name: 'Tolko Industries', url: 'https://tolko.com/divisions/corporate-office/' },
    { name: 'Vernon Seed Orchard Company', url: 'https://www.vsoc.ca/' }
];

async function captureScreenshot(url, filename) {
    const browser = await puppeteer.launch({ 
        headless: true,
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    
    try {
        const page = await browser.newPage();
        await page.setViewport({ width: 1200, height: 800 });
        
        console.log(`Capturing: ${url}`);
        await page.goto(url, { 
            waitUntil: 'networkidle2', 
            timeout: 30000 
        });
        
        // Wait a bit for any dynamic content to load
        await new Promise(resolve => setTimeout(resolve, 3000));
        
        // Capture full page screenshot first
        const fullScreenshot = await page.screenshot({ 
            fullPage: true,
            type: 'png'
        });
        
        console.log(`✓ Full page captured for: ${url}`);
        
        // Now we need to resize the image to 400x400
        // Save the full screenshot temporarily
        const tempFilename = filename.replace('.png', '_temp.png');
        fs.writeFileSync(tempFilename, fullScreenshot);
        
        // Use sharp to resize the image
        await sharp(tempFilename)
            .resize(400, 400, {
                fit: 'cover',
                position: 'top'
            })
            .png()
            .toFile(filename);
        
        // Remove temporary file
        fs.unlinkSync(tempFilename);
        
        console.log(`✓ Resized screenshot saved: ${filename}`);
    } catch (error) {
        console.error(`✗ Failed to capture ${url}:`, error.message);
    } finally {
        await browser.close();
    }
}

async function captureAllScreenshots() {
    // Create screenshots directory
    const screenshotsDir = './screenshots';
    if (!fs.existsSync(screenshotsDir)) {
        fs.mkdirSync(screenshotsDir);
    }
    
    console.log(`Starting to capture ${urls.length} screenshots...`);
    
    for (let i = 0; i < urls.length; i++) {
        const { name, url } = urls[i];
        const filename = path.join(screenshotsDir, `${name.replace(/[^a-zA-Z0-9]/g, '_').toLowerCase()}.png`);
        
        try {
            await captureScreenshot(url, filename);
        } catch (error) {
            console.error(`Failed to process ${name}:`, error.message);
        }
        
        // Add delay between requests to be respectful
        if (i < urls.length - 1) {
            await new Promise(resolve => setTimeout(resolve, 2000));
        }
    }
    
    console.log('Screenshot capture complete!');
}

captureAllScreenshots().catch(console.error);
