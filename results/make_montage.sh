#!/bin/bash

# montage is a program from ImageMagick

montage -tile 5x5 -geometry 800x600 \
"dead_grid_stat_[0][0].png" \
"dead_grid_stat_[0][1].png" \
"dead_grid_stat_[0][2].png" \
"dead_grid_stat_[0][3].png" \
"dead_grid_stat_[0][4].png" \
"dead_grid_stat_[1][0].png" \
"dead_grid_stat_[1][1].png" \
"dead_grid_stat_[1][2].png" \
"dead_grid_stat_[1][3].png" \
"dead_grid_stat_[1][4].png" \
"dead_grid_stat_[2][0].png" \
"dead_grid_stat_[2][1].png" \
"dead_grid_stat_[2][2].png" \
"dead_grid_stat_[2][3].png" \
"dead_grid_stat_[2][4].png" \
"dead_grid_stat_[3][0].png" \
"dead_grid_stat_[3][1].png" \
"dead_grid_stat_[3][2].png" \
"dead_grid_stat_[3][3].png" \
"dead_grid_stat_[3][4].png" \
"dead_grid_stat_[4][0].png" \
"dead_grid_stat_[4][1].png" \
"dead_grid_stat_[4][2].png" \
"dead_grid_stat_[4][3].png" \
"dead_grid_stat_[4][4].png"  \
montage.png
