#!/bin/bash 
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -t 0-04:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p huce_cascade,shared,sapphire       # Partition to submit to
#SBATCH --mem=1000


IC_PATH="/n/home04/sweidman/holylfs06/IC_CESM2/"
RUN_MEAN_CASE="nudge30_coef_1"
TEND_OUT_CASE="nudge30_coef_1_i10"
NUDGE_PATH="/n/home04/sweidman/holylfs06/CESM2_corrector_out/Run/archive/${RUN_MEAN_CASE}/atm/hist/"

tau=21600
cpair=1.00464e3 

# ----- years to average over -----
# Edit this list for whatever set of years you want
#years=(2000 2001 2002 2003 2004 2005 2006 2007 2008 2009)
years=($(seq 1980 1 2009))

for month in {1..12}; do
  files_mon=()
  for day in {1..31}; do
    for which in {0..3..1}; do

      monthno=$(printf '%02d' $month)
      dayno=$(printf '%02d' $day)
      hourno=$(printf '%05d' $((which*21600)))

      # --- NEW: shift to next timestep for input selection (year wraps) ---
      ts_next=$((which+1))
      in_monthno=$monthno
      in_dayno=$dayno
      if [ $ts_next -eq 4 ]; then
        ts_next=0
        if next_md=$(date -u -d "2001-${monthno}-${dayno} + 1 day" +%m-%d 2>/dev/null); then
          in_monthno=${next_md%-*}
          in_dayno=${next_md#*-}
        else
          continue  # invalid calendar date (e.g. Apr 31)
        fi  
      fi
      # ---------------------------------------------------------------

      OUT="${IC_PATH}${TEND_OUT_CASE}.${monthno}-${dayno}-${hourno}.nc"

      if [ ! -f $OUT ] ; then
        # Build list of files for all years for this day
        files=()
        for yr in "${years[@]}"; do
          yrno=$(printf "%04d" $yr)
          f="${NUDGE_PATH}${RUN_MEAN_CASE}.cam.h2.${yrno}-${in_monthno}-${in_dayno}-00000.nc"
          if [ -f $f ] ; then
            files+=("$f")
          else
            echo "no file $f"
          fi
        done

        # Use the first year’s file to query time dimension length
        run_mean_file="${files[0]}"
        echo "Processing files:"
        printf '  %s\n' "${files[@]}"

        tmp1="${IC_PATH}/tmp/tmp1_${RUN_MEAN_CASE}.nc"
        tmp2="${IC_PATH}/tmp/tmp2_${RUN_MEAN_CASE}.nc"
        tmp3="${IC_PATH}/tmp/tmp3_${RUN_MEAN_CASE}.nc"

        function ncdmnsz { ncks --trd -m -M ${2} | grep -E -i ": ${1}, size =" | cut -f 7 -d ' ' | uniq ; }
        len_time=$( ncdmnsz time "$run_mean_file" )

        # --- Take multi-year mean at the chosen time index ---

        # ncea: ensemble average across files
        # -d time,<idx> slices each file BEFORE averaging,
        # so we end up with a single-record time dimension.
        if [ "$len_time" -eq 4 ]; then
          ncea -O -d time,$ts_next "${files[@]}" "$tmp1"
        elif [ "$len_time" -eq 8 ]; then
          ncea -O -d time,$((ts_next*2)) "${files[@]}" "$tmp1"
        else
          echo "Unexpected time dimension length ($len_time) in $run_mean_file, skipping"
          continue
        fi

        # Create scaled, renamed vars from multi-year mean
        ncap2 -O -s "UDIFF=${tau}*Running_nudge_U(0,:,:,:); \
            VDIFF=${tau}*Running_nudge_V(0,:,:,:); \
            QDIFF=${tau}*Running_nudge_Q(0,:,:,:); \
            SDIFF=${tau}*${cpair}*Running_nudge_T(0,:,:,:);" "$tmp1" "$tmp2"

        # Drop unused vars/time
        ncks -O -v UDIFF,VDIFF,QDIFF,SDIFF,lat,lev,lon "$tmp2" "$tmp3"
        ncks -O -C -x -v hyai,hybi,hyam,hybm,ilev,P0,PS "$tmp3" "$tmp3"

        # Permute to (lev,lat,lon)
        ncpdq -O -a lev,lat,lon "$tmp3" "$OUT"

        # Clean coordinates attributes
        ncatted -O \
          -a coordinates,UDIFF,d,, \
          -a coordinates,VDIFF,d,, \
          -a coordinates,QDIFF,d,, \
          -a coordinates,SDIFF,d,, \
          "$OUT"

        rm -f "$tmp1" "$tmp2" "$tmp3"
        echo "Wrote $OUT"
        files_mon+=("$OUT")
      fi
    done
  done

  # make monthly average file
  if [ ! -f ${IC_PATH}/monave/${TEND_OUT_CASE}.${monthno}.nc ] ; then
  nces ${files_mon[@]} ${IC_PATH}/monave/${TEND_OUT_CASE}.${monthno}.nc
  fi

  # if only need to take the monthly average, use this line in bash:
  # for mon in {01..12}; do nces <case>.${mon}* monave/<case>.${mon}.nc; done 
done

