#!/bin/bash
# batch_process_zarr.sh - Process multiple Zarr datasets with both process_zarr.py and process_newvis.py

# Configuration options for process_zarr.py
VIS_TYPE="all"  # Visualization types to generate
MIN_PIXELS=1000 # Minimum pixels to include a class
TOP_N=15        # Number of top classes to display

# Configuration options for process_newvis.py
NEWVIS_TYPE="stacked_bar"  # stacked_bar or treemap
SECVIS_TYPE="treemap"  # Secondary visualization type
MIN_TRANSITION_PCT=1.0     # Minimum percentage for a transition to be shown
NEWVIS_TOP_N=10           # Number of top classes to display for transitions

OUTPUT_ROOT="./visualization_results"

# Create output root directory
mkdir -p "$OUTPUT_ROOT"

# Log file setup
LOG_FILE="$OUTPUT_ROOT/batch_processing.log"
echo "Batch processing started at $(date)" > "$LOG_FILE"

# Function to process a single Zarr dataset
process_zarr_dataset() {
    local zarr_path="$1"
    
    # Extract region name and grid name for organized output
    region_name=$(echo "$zarr_path" | cut -d'/' -f2)
    grid_name=$(echo "$zarr_path" | cut -d'/' -f3)
    
    # Create output directory with region and grid name
    output_dir="$OUTPUT_ROOT/${region_name}/${grid_name}"
    mkdir -p "$output_dir"
    
    echo "Processing: $zarr_path" | tee -a "$LOG_FILE"
    echo "Output to: $output_dir" | tee -a "$LOG_FILE"
    
    # Run process_zarr.py for general persistence visualizations
    echo "Running process_zarr.py..." | tee -a "$LOG_FILE"
    python ../process_zarr.py "$zarr_path" \
        --output_dir "$output_dir" \
        --vis_type "$VIS_TYPE" \
        --min_pixels "$MIN_PIXELS" \
        --top_n "$TOP_N"
    
    zarr_result=$?
    
    # Run process_newvis.py for transition visualizations
    echo "Running process_newvis.py..." | tee -a "$LOG_FILE"
    python ../process_newvis.py "$zarr_path" \
        --output_dir "$output_dir" \
        --vis_type "$NEWVIS_TYPE" \
        --min_pixels "$MIN_PIXELS" \
        --min_transition_pct "$MIN_TRANSITION_PCT" \
        --top_n "$NEWVIS_TOP_N"
    
    newvis_result=$?

    # Run process_newvis.py for transition visualizations
        echo "Running process_newvis.py..." | tee -a "$LOG_FILE"
        python ../process_newvis.py "$zarr_path" \
            --output_dir "$output_dir" \
            --vis_type "$SECVIS_TYPE" \
            --min_pixels "$MIN_PIXELS" \
            --min_transition_pct "$MIN_TRANSITION_PCT" \
            --top_n "$NEWVIS_TOP_N"
        
        newvis_result=$?

    
    
    # Check if processing was successful
    if [ $zarr_result -eq 0 ] && [ $newvis_result -eq 0 ]; then
        echo "✓ Success: $zarr_path (both visualizations)" | tee -a "$LOG_FILE"
        return 0
    elif [ $zarr_result -eq 0 ]; then
        echo "⚠ Partial success: $zarr_path (only process_zarr.py succeeded)" | tee -a "$LOG_FILE"
        return 1
    elif [ $newvis_result -eq 0 ]; then
        echo "⚠ Partial success: $zarr_path (only process_newvis.py succeeded)" | tee -a "$LOG_FILE"
        return 1
    else
        echo "✗ Failed: $zarr_path (both visualizations failed)" | tee -a "$LOG_FILE"
        return 2
    fi
}

# List of all Zarr datasets to process
datasets=(
"./BA_Nordeste_Baiano_13/grid_39.508400W_37.399033W_12.408242S_11.439342S/data.zarr"
"./SE_Agreste_Sergipano_10/grid_37.793115W_36.900749W_11.174317S_10.163120S/data.zarr"
"./ES_Sul_Espírito-santense_31/grid_41.879796W_40.801002W_21.285535S_20.197775S/data.zarr"
"./RN_Leste_Potiguar_1/grid_35.549500W_34.986600W_6.558393S_5.154100S/data.zarr"
"./PR_Norte_Central_Paranaense_55/grid_52.421974W_50.875576W_24.970999S_22.539513S/data.zarr"
"./SC_Sul_Catarinense_68/grid_50.177509W_48.635306W_29.355066S_27.935931S/data.zarr"
"./PR_Oeste_Paranaense_58/grid_54.563000W_52.584410W_25.651392S_23.981406S/data.zarr"
"./RJ_Sul_Fluminense_36/grid_44.889321W_43.400414W_23.334447S_22.051534S/data.zarr"
"./PR_Sudoeste_Paranaense_59/grid_53.991038W_52.213996W_26.416062S_25.476538S/data.zarr"
"./SC_Grande_Florianópolis_67/grid_49.510440W_48.394400W_28.108454S_27.134336S/data.zarr"
"./SE_Sertão_Sergipano_9/grid_37.562400W_37.000051W_10.645799S_9.919466S/data.zarr"
"./SP_Piracicaba_43/grid_47.959056W_47.167600W_23.223234S_22.063582S/data.zarr"
"./SC_Oeste_Catarinense_63/grid_53.837149W_50.484819W_27.531204S_26.241614S/data.zarr"
"./ES_Litoral_Norte_Espírito-santense_29/grid_40.662037W_39.689521W_20.048790S_17.891945S/data.zarr"
"./SC_Serrana_65/grid_51.632579W_49.280721W_28.631630S_26.648227S/data.zarr"
"./MG_Vale_do_Rio_Doce_23/grid_43.294984W_40.907693W_20.206102S_17.766405S/data.zarr"
"./SP_Presidente_Prudente_45/grid_53.110054W_50.609470W_22.697179S_21.050982S/data.zarr"
"./PR_Norte_Pioneiro_Paranaense_56/grid_51.010402W_49.548433W_24.122595S_22.815505S/data.zarr"
"./MG_Zona_da_Mata_27/grid_44.222965W_41.411140W_22.233246S_19.828153S/data.zarr"
"./SP_Metropolitana_de_São_Paulo_52/grid_47.208522W_45.694814W_24.089731S_23.183419S/data.zarr"
"./PR_Noroeste_Paranaense_53/grid_54.169008W_51.996905W_24.260962S_22.516295S/data.zarr"
"./BA_Centro_Norte_Baiano_12/grid_39.695892W_38.520765W_13.022239S_11.787900S/data.zarr"
"./SC_Vale_do_Itajaí_66/grid_50.349349W_48.503000W_27.663257S_26.487776S/data.zarr"
"./SP_Assis_47/grid_51.287779W_49.260200W_23.541123S_22.019328S/data.zarr"
"./MG_Oeste_de_Minas_24/grid_45.541500W_44.299480W_21.235858S_19.709870S/data.zarr"
"./MG_Jequitinhonha_18/grid_43.705672W_39.856829W_18.718229S_15.646007S/data.zarr"
"./PB_Agreste_Paraibano_2/grid_35.576200W_35.263561W_7.502713S_6.901688S/data.zarr"
"./SP_Araçatuba_40/grid_51.767732W_49.822800W_21.725785S_20.296881S/data.zarr"
"./SP_Macro_Metropolitana_Paulista_49/grid_48.207462W_46.041000W_24.135093S_22.767617S/data.zarr"
"./RJ_Centro_Fluminense_34/grid_43.436994W_41.702600W_22.457834S_21.697209S/data.zarr"
"./MG_Norte_de_Minas_17/grid_42.792200W_41.327265W_16.495025S_14.930576S/data.zarr"
"./PR_Centro_Oriental_Paranaense_57/grid_51.374457W_49.235461W_25.651964S_23.795561S/data.zarr"
"./SE_Leste_Sergipano_11/grid_37.866849W_36.426597W_11.568590S_10.035830S/data.zarr"
"./RS_Metropolitana_de_Porto_Alegre_73/grid_52.232612W_49.774122W_29.951800S_29.198855S/data.zarr"
"./RS_Sudoeste_Rio-grandense_74/grid_55.436200W_54.862721W_29.469800S_28.087818S/data.zarr"
"./RJ_Baixadas_35/grid_42.678955W_41.879933W_22.979339S_22.351895S/data.zarr"
"./ES_Central_Espírito-santense_30/grid_41.416110W_29.299456W_20.892984S_19.559363S/data.zarr"
"./SP_Itapetininga_48/grid_49.610737W_47.573454W_24.726202S_22.860668S/data.zarr"
"./SP_Campinas_44/grid_47.530549W_46.332566W_23.226034S_21.281675S/data.zarr"
"./SC_Norte_Catarinense_64/grid_51.270363W_48.517200W_26.863078S_25.955842S/data.zarr"
"./GO_Sul_Goiano_77/grid_51.171500W_48.342900W_19.498364S_17.933700S/data.zarr"
"./AL_Agreste_Alagoano_7/grid_37.127059W_36.191656W_10.071094S_9.212100S/data.zarr"
"./PE_Mata_Pernambucana_5/grid_36.125488W_34.836558W_8.917152S_7.374780S/data.zarr"
"./AL_Leste_Alagoano_8/grid_36.836583W_35.156756W_10.435500S_8.813127S/data.zarr"
"./MG_Triângulo_Mineiro/Alto_Paranaíba_20/grid_51.023400W_47.925900W_20.440810S_18.334800S/data.zarr"
"./BA_Metropolitana_de_Salvador_14/grid_39.505789W_37.891650W_13.259674S_12.134537S/data.zarr"
"./PR_Centro-Sul_Paranaense_60/grid_53.152572W_50.960103W_26.599363S_24.449266S/data.zarr"
"./RJ_Metropolitana_do_Rio_de_Janeiro_37/grid_44.193108W_42.441784W_23.080018S_22.084263S/data.zarr"
"./PE_Agreste_Pernambucano_4/grid_36.696800W_35.304459W_9.243772S_7.514566S/data.zarr"
"./PB_Mata_Paraibana_3/grid_35.401583W_34.823200W_7.549158S_6.497678S/data.zarr"
"./MG_Sul/Sudoeste_de_Minas_25/grid_46.972200W_43.976927W_22.922755S_20.559800S/data.zarr"
"./SP_Vale_do_Paraíba_Paulista_50/grid_46.263659W_44.161365W_23.939000S_22.403848S/data.zarr"
"./SP_Bauru_41/grid_50.274220W_47.927466W_23.429369S_21.318782S/data.zarr"
"./BA_Centro_Sul_Baiano_15/grid_42.644552W_39.122136W_16.005561S_12.874571S/data.zarr"
"./RS_Noroeste_Rio-grandense_69/grid_55.431432W_51.297792W_29.367705S_27.082302S/data.zarr"
"./MG_Central_Mineira_21/grid_45.488991W_44.975054W_20.206765S_19.579032S/data.zarr"
"./MG_Campo_das_Vertentes_26/grid_45.399468W_43.417235W_21.730487S_20.711118S/data.zarr"
"./MG_Vale_do_Mucuri_19/grid_42.288231W_40.174127W_18.502476S_16.781811S/data.zarr"
"./RJ_Norte_Fluminense_33/grid_42.264173W_40.993300W_22.439251S_21.195588S/data.zarr"
"./PE_Metropolitana_de_Recife_6/grid_35.265968W_32.452100W_8.571381S_3.864233S/data.zarr"
"./SP_Marília_46/grid_50.864662W_49.422776W_22.505600S_21.656767S/data.zarr"
"./SP_São_José_do_Rio_Preto_38/grid_51.087431W_48.691924W_21.659100S_19.788900S/data.zarr"
"./MS_Leste_de_Mato_Grosso_do_Sul_75/grid_53.766102W_50.927398W_22.987596S_19.435159S/data.zarr"
"./ES_Noroeste_Espírito-santense_28/grid_41.245477W_40.124584W_19.805687S_17.952047S/data.zarr"
"./RN_Agreste_Potiguar_0/grid_35.370000W_35.298426W_6.426946S_6.048249S/data.zarr"
"./BA_Sul_Baiano_16/grid_40.623475W_38.901300W_18.320467S_13.198933S/data.zarr"
"./MS_Sudoeste_de_Mato_Grosso_do_Sul_76/grid_55.663287W_53.482914W_24.056854S_21.577900S/data.zarr"
"./SP_Araraquara_42/grid_49.241217W_47.510158W_21.921122S_21.487291S/data.zarr"
"./PR_Sudeste_Paranaense_61/grid_51.821222W_49.964312W_26.717122S_24.752944S/data.zarr"
"./MG_Metropolitana_de_Belo_Horizonte_22/grid_45.000414W_42.474462W_20.927592S_18.068036S/data.zarr"
"./PR_Metropolitana_de_Curitiba_62/grid_50.231520W_48.023537W_26.237353S_24.416949S/data.zarr"
"./SP_Litoral_Sul_Paulista_51/grid_48.600831W_46.589077W_25.311367S_23.937847S/data.zarr"
"./SP_Ribeirão_Preto_39/grid_48.894441W_47.075881W_21.808205S_20.438100S/data.zarr"
"./RS_Nordeste_Rio-grandense_70/grid_52.329545W_49.691352W_29.560746S_27.719463S/data.zarr"
"./RS_Centro_Ocidental_Rio-grandense_71/grid_54.915984W_53.028481W_29.836552S_29.126206S/data.zarr"
"./PR_Centro_Ocidental_Paranaense_54/grid_53.289738W_51.889528W_24.932012S_23.552491S/data.zarr"
"./RS_Centro_Oriental_Rio-grandense_72/grid_53.287406W_51.647075W_29.898900S_29.014512S/data.zarr"
"./RJ_Noroeste_Fluminense_32/grid_42.369892W_41.469630W_21.861044S_20.763205S/data.zarr"
)

# Process each dataset
echo "Starting batch processing of ${#datasets[@]} Zarr datasets..." | tee -a "$LOG_FILE"
echo "-------------------------------------------" | tee -a "$LOG_FILE"

total_datasets=${#datasets[@]}
fully_successful=0
partially_successful=0
failed=0

for dataset in "${datasets[@]}"; do
    # Process the dataset
    process_zarr_dataset "$dataset"
    result=$?
    
    # Update counters based on success/failure
    if [ $result -eq 0 ]; then
        ((fully_successful++))
    elif [ $result -eq 1 ]; then
        ((partially_successful++))
    else
        ((failed++))
    fi
    
    echo "-------------------------------------------" | tee -a "$LOG_FILE"
done

# Print summary
echo "Batch processing completed at $(date)" | tee -a "$LOG_FILE"
echo "Total datasets: $total_datasets" | tee -a "$LOG_FILE"
echo "Fully successful: $fully_successful" | tee -a "$LOG_FILE"
echo "Partially successful: $partially_successful" | tee -a "$LOG_FILE"
echo "Failed: $failed" | tee -a "$LOG_FILE"
echo "All results saved to: $OUTPUT_ROOT" | tee -a "$LOG_FILE"

# Make the script executable with: chmod +x batch_process_zarr.sh