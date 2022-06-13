file(REMOVE_RECURSE
  "libnvinfer.pdb"
  "libnvinfer.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/nvinfer.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
