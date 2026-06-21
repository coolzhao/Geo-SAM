# Geo-SAM translations

The source language is English. Translation catalogs use the two-letter locale
suffix expected by `tools/i18n.py`.

Regenerate the TS catalogs from the plugin root with:

```shell
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate qgis39
pylupdate5 __init__.py geo_sam_tool.py geo_sam_provider.py tools/*.py ui/*.py \
  ui/*.ui -ts i18n/GeoSAM_{zh,ja,ko,fr,de,ru,ar,es,pt}.ts
```

Compile runtime catalogs with:

```shell
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate qgis39
lrelease i18n/*.ts
```
