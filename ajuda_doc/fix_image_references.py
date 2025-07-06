import re

# Read the file
with open('forestry_indigenous_sort.md', 'r') as f:
    content = f.read()

# Define the mappings from current image names to actual filenames
image_mappings = {
    'westbank_first_nation.png': 'westbank_first_nation.png',
    'okanagan_indian_band.png': 'okanagan_indian_band.png',
    'adams_lake_indian_band.png': 'adams_lake_indian_band.png',
    'lower_similkameen_indian_band.png': 'lower_similkameen_indian_band.png',
    'penticton_indian_band.png': 'penticton_indian_band.png',
    'upper_nicola_band.png': 'upper_nicola_band.png',
    'tsideldel_first_nation.png': 'tsideldel_first_nation.png',
    'splatsin_first_nation.png': 'splatsin_first_nation.png',
    'esketemc_first_nation.png': 'esketemc_first_nation.png',
    'alkali_resource_management.png': 'alkali_resource_management.png',
    'okanagan_nation_alliance.png': 'okanagan_nation_alliance.png',
    'okanagan_training_and_consulting.png': 'okanagan_training_and_consulting.png',
    'sd73_aboriginal_education.png': 'sd73_aboriginal_education.png',
    'kitaskinaw_education_authority.png': 'kitaskinaw_education_authority.png',
    'first_nations_summit.png': 'first_nations_summit.png',
    'union_of_bc_indian_chiefs.png': 'union_of_bc_indian_chiefs.png',
    'neskonlith_indian_band.png': 'neskonlith_indian_band.png',
    'bc_ministry_of_forests.png': 'bc_ministry_of_forests.png',
    'bc_forestry_innovation_investment.png': 'bc_forestry_innovation_investment.png',
    'council_of_forest_industries.png': 'council_of_forest_industries.png',
    'first_nations_forestry_council.png': 'first_nations_forestry_council.png',
    'ubc_okanagan_indigenous_initiatives.png': 'ubc_okanagan_indigenous_initiatives.png',
    'selkirk_college_aric.png': 'selkirk_college_aric.png',
    'thompson_rivers_university.png': 'thompson_rivers_university.png',
    'university_of_victoria.png': 'university_of_victoria.png',
    'indigenous_services_canada.png': 'indigenous_services_canada.png',
    'natural_resources_canada.png': 'natural_resources_canada.png',
    'crown_indigenous_relations_canada.png': 'crown_indigenous_relations_canada.png',
    'environment_and_climate_change_canada.png': 'environment_and_climate_change_canada.png',
    'ecojustice.png': 'ecojustice.png',
    'cpaws_bc.png': 'cpaws_bc.png',
    'yellowstone_to_yukon.png': 'yellowstone_to_yukon.png',
    'tolko_industries.png': 'tolko_industries.png',
    'vernon_seed_orchard_company.png': 'vernon_seed_orchard_company.png',
    'mercer.png': 'mercer.png',
    'forest_professionals_bc.png': 'forest_professionals_bc.png',
    'bc_community_forest_association.png': 'bc_community_forest_association.png'
}

# Replace all image references with the new format
# Pattern: ![alt text](/long/path/to/filename.png)
# Replace with: ![alt text](<filename.png>)
for old_filename, new_filename in image_mappings.items():
    # Match the current pattern with absolute path
    pattern = r'!\[([^\]]+)\]\(/media/bndt/db_lin/gitstore/leandromet/nlp_project_cuda/ajuda_doc/screenshots/' + re.escape(old_filename) + r'\)'
    replacement = r'![\1](<' + new_filename + r'>)'
    content = re.sub(pattern, replacement, content)

# Write the file back
with open('forestry_indigenous_sort.md', 'w') as f:
    f.write(content)

print("All image references updated to use angle bracket format!")
