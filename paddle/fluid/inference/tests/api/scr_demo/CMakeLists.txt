set(SCR_DEMO_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/scr_demo")
inference_analysis_test_build(test_analyzer_scr_demo SRCS analyzer_resnet50_trainer_tester.cc
  EXTRA_DEPS ${INFERENCE_EXTRA_DEPS})
ADD_CUSTOM_COMMAND(
	TARGET test_analyzer_scr_demo POST_BUILD
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/main_program  ${SCR_DEMO_INSTALL_DIR}
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/startup_program  ${SCR_DEMO_INSTALL_DIR}
        COMMENT "copying *_program -> ${SCR_DEMO_INSTALL_DIR}")
