# ==============================
# Shaders
# ==============================
from OpenGL.GL import *

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0);
    gl_PointSize = 2.0;
}
"""

FRAGMENT_SHADER = """
#version 330 core
uniform vec4 material_color;
out vec4 FragColor;
void main()
{
    FragColor = material_color;
}
"""

# Point cloud: per-vertex color (white-to-blue scale)
VERTEX_SHADER_POINTS = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float point_size;

out vec3 vColor;

void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0);
    gl_PointSize = point_size;
    vColor = color;
}
"""

FRAGMENT_SHADER_POINTS = """
#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main()
{
    FragColor = vec4(vColor, 1.0);
}
"""

def create_shader_program():
    v = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(v, VERTEX_SHADER)
    glCompileShader(v)

    f = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(f, FRAGMENT_SHADER)
    glCompileShader(f)

    program = glCreateProgram()
    glAttachShader(program, v)
    glAttachShader(program, f)
    glLinkProgram(program)

    glDeleteShader(v)
    glDeleteShader(f)
    return program


def create_pointcloud_shader_program():
    v = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(v, VERTEX_SHADER_POINTS)
    glCompileShader(v)

    f = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(f, FRAGMENT_SHADER_POINTS)
    glCompileShader(f)

    program = glCreateProgram()
    glAttachShader(program, v)
    glAttachShader(program, f)
    glLinkProgram(program)

    glDeleteShader(v)
    glDeleteShader(f)
    return program