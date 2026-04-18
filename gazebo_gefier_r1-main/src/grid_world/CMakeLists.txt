cmake_minimum_required(VERSION 3.8)
project(grid_world)

find_package(ament_cmake REQUIRED)

# This is the magic part that exports your folders to the install directory
install(DIRECTORY
  worlds
  materials
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
