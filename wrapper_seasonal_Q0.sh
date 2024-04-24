source /storage/home/nargessayah/narges_scripts/myenv/bin/activate

pip install pyyaml

nohup python seasonal_drought.py >nohupQ0SeasonalDrought.out -n BC -read /storage/data/projects/hydrology/vic_gen2/output/CanESM2-LE/fraser/init2/MOUTH -write /storage/home/nargessayah/narges_env/outputs/drought/

echo "Python script started in background. Console output will be logged in nohupQ0SeasonalDrought.out"