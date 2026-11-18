#!/bin/sh
# Downloads the complete GeoNames file and keeps only what the place list uses:
# populated places, second/third-level districts, islands and named regions.
#   -> build/work/geonames/places_all.tsv  (id, name, alternate names, class, code, country, population)
cd "$(dirname "$0")/work/geonames" || exit 1
curl -sS -L --retry 3 https://download.geonames.org/export/dump/allCountries.zip -o ac.zip || exit 1
unzip -p ac.zip allCountries.txt | awk -F"\t" '($7=="P") || ($7=="A" && ($8=="ADM2" || $8=="ADM3")) || ($7=="T" && ($8=="ISL" || $8=="ISLS")) || ($7=="L" && $8=="RGN") {print $1"\t"$2"\t"$4"\t"$7"\t"$8"\t"$9"\t"$15}' > places_all.tsv
touch ac.done
