cat > add_all_images.py << 'EOF'
import re

# Read the file
with open('forestry_indigenous_sort.md', 'r') as f:
    content = f.read()

# Define the mappings from URLs to image files
url_to_image = {
    'https://www.wfn.ca': 'westbank_first_nation.png',
    'https://www.okib.ca': 'okanagan_indian_band.png',
    'https://www.lsib.net': 'lower_similkameen_indian_band.png',
    'https://www.pib.ca': 'penticton_indian_band.png',
    'https://uppernicola.com': 'upper_nicola_band.png',
    'https://www.tsideldel.org/': 'tsideldel_first_nation.png',
    'http://www.splatsin.ca/': 'splatsin_first_nation.png',
    'https://www.esketemc.ca/': 'esketemc_first_nation.png',
    'https://www.armltd.org/': 'alkali_resource_management.png',
    'https://www.syilx.org': 'okanagan_nation_alliance.png',
    'https://okcp.ca': 'okanagan_training_and_consulting.png',
    'https://dallas.sd73.bc.ca/en/our-schools-programs/aboriginal-education.aspx': 'sd73_aboriginal_education.png',
    'https://kitaskinaw.com/': 'kitaskinaw_education_authority.png',
    'https://fns.bc.ca/': 'first_nations_summit.png',
    'https://www.ubcic.bc.ca/': 'union_of_bc_indian_chiefs.png',
    'https://neskonlith.net/': 'neskonlith_indian_band.png',
    'https://www2.gov.bc.ca/gov/content/governments/government-structure/ministries-organizations/ministries/forests': 'bc_ministry_of_forests.png',
    'https://www.bcfii.ca': 'bc_forestry_innovation_investment.png',
    'https://www.cofi.org': 'council_of_forest_industries.png',
    'https://www.fnforestrycouncil.ca': 'first_nations_forestry_council.png',
    'https://equity.ok.ubc.ca/indigenous-initiatives/': 'ubc_okanagan_indigenous_initiatives.png',
    'https://selkirk.ca/ari': 'selkirk_college_aric.png',
    'https://www.tru.ca/nrs/': 'thompson_rivers_university.png',
    'https://www.uvic.ca/socialsciences/environmental/': 'university_of_victoria.png',
    'https://www.sac-isc.gc.ca': 'indigenous_services_canada.png',
    'https://natural-resources.canada.ca': 'natural_resources_canada.png',
    'https://www.rcaanc-cirnac.gc.ca/': 'crown_indigenous_relations_canada.png',
    'https://www.canada.ca/en/environment-climate-change.html': 'environment_and_climate_change_canada.png',
    'https://ecojustice.ca': 'ecojustice.png',
    'https://cpawsbc.org': 'cpaws_bc.png',
    'https://y2y.net': 'yellowstone_to_yukon.png',
    'https://tolko.com/divisions/corporate-office/': 'tolko_industries.png',
    'https://www.vsoc.ca/': 'vernon_seed_orchard_company.png',
    'https://mercerint.com': 'mercer.png',
    'https://www.fpbc.ca/': 'forest_professionals_bc.png',
    'https://bccfa.ca/community-forests-and-value-added-enterprises/': 'bc_community_forest_association.png'
}

# For each URL, add the image after it
for url, image_file in url_to_image.items():
    # Escape special characters in URL for regex
    escaped_url = re.escape(url)
    # Find the URL and add the image after it
    pattern = f'({escaped_url})'
    replacement = f'\\1\n\n        ![{image_file.replace("_", " ").replace(".png", "").title()}](screenshots/{image_file})'
    content = re.sub(pattern, replacement, content)

# Write the file back
with open('forestry_indigenous_sort.md', 'w') as f:
    f.write(content)

print("All images added successfully!")
EOF

python3 add_all_images.py