{
  "targets": [
    {
      "target_name": "litert_lm_node",
      "sources": [
        "src/native/litert_addon.cc"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "<(module_root_dir)/prebuilt/include"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc!": [ "-fno-exceptions" ],
      "defines": [ "NAPI_DISABLE_CPP_EXCEPTIONS" ],
      "conditions": [
        [
          "OS=='mac'",
          {
            "xcode_settings": {
              "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
              "CLANG_CXX_LIBRARY": "libc++",
              "MACOSX_DEPLOYMENT_TARGET": "10.15",
              "LD_RUNPATH_SEARCH_PATHS": [
                "@loader_path"
              ]
            },
            "libraries": [
              "-L<(module_root_dir)/prebuilt/darwin/lib",
              "-llitert_lm_rust_api",
              "-lLiteRtRuntimeCApi"
            ],
            "link_settings": {
              "library_dirs": [
                "<(module_root_dir)/prebuilt/darwin/lib"
              ]
            }
          }
        ],
        [
          "OS=='linux'",
          {
            "cflags_cc": [
              "-std=c++17",
              "-fexceptions"
            ],
            "libraries": [
              "-L<(module_root_dir)/prebuilt/linux/lib",
              "-llitert_lm_rust_api",
              "-lLiteRtRuntimeCApi"
            ],
            "link_settings": {
              "library_dirs": [
                "<(module_root_dir)/prebuilt/linux/lib"
              ],
              "ldflags": [
                "-Wl,-rpath,$$ORIGIN"
              ]
            }
          }
        ],
        [
          "OS=='win'",
          {
            "libraries": [
              "<(module_root_dir)/prebuilt/windows/lib/litert_lm_rust_api.lib",
              "<(module_root_dir)/prebuilt/windows/lib/LiteRtRuntimeCApi.lib"
            ]
          }
        ]
      ]
    }
  ]
}
