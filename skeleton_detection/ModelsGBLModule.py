import numpy as np
from OpenGL.GL import *
from pygltflib import GLTF2
from PIL import Image
import ctypes
import io
import os


class PlyMesh:
    def __init__(self):
        self.type = -1

        self.vao = 0
        self.vbo = 0
        self.ebo = 0

        self.vertex_count = 0
        self.index_count = 0

        self.texture_id = 0
        self.has_texture = False

        self.base_color = np.array([1, 1, 1, 1], dtype=np.float32)

        self.metallic_factor = 0.0
        self.roughness_factor = 1.0

        self.local_transform = np.eye(4, dtype=np.float32)

        self.frame_id = "map"
        self.model_name = ""


class ModelUpload:

    def __init__(self):

        self.view_matrix = np.eye(4, dtype=np.float32)
        self.proj_matrix = np.eye(4, dtype=np.float32)

    # ---------------------------
    # Shader
    # ---------------------------

    def create_glb_shader(self):

        vertex = """
        #version 330 core

        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 normal;
        layout(location = 2) in vec2 tex_coord;

        uniform mat4 model_matrix;
        uniform mat4 view_matrix;
        uniform mat4 projection_matrix;

        out vec3 frag_pos;
        out vec3 frag_normal;
        out vec2 frag_uv;

        void main()
        {
            vec4 world = model_matrix * vec4(position,1.0);

            frag_pos = world.xyz;

            mat3 normal_matrix = mat3(transpose(inverse(model_matrix)));
            frag_normal = normal_matrix * normal;

            frag_uv = tex_coord;

            gl_Position = projection_matrix * view_matrix * world;
        }
        """

        fragment = """
        #version 330 core

        in vec3 frag_pos;
        in vec3 frag_normal;
        in vec2 frag_uv;

        uniform vec4 base_color;
        uniform sampler2D base_color_texture;
        uniform bool has_texture;

        out vec4 FragColor;

        void main()
        {
            vec4 color = base_color;

            if(has_texture)
                color *= texture(base_color_texture, frag_uv);

            vec3 light_dir = normalize(vec3(0.5,1.0,0.8));
            vec3 normal = normalize(frag_normal);

            float diff = max(dot(normal, light_dir), 0.0);

            vec3 ambient = 0.3 * color.rgb;
            vec3 diffuse = diff * color.rgb;

            FragColor = vec4(ambient + diffuse, color.a);
        }
        """

        return self._compile_shader(vertex, fragment)

    def _compile_shader(self, vsrc, fsrc):

        v = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(v, vsrc)
        glCompileShader(v)

        if not glGetShaderiv(v, GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(v))

        f = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(f, fsrc)
        glCompileShader(f)

        if not glGetShaderiv(f, GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(f))

        program = glCreateProgram()

        glAttachShader(program, v)
        glAttachShader(program, f)

        glLinkProgram(program)

        if not glGetProgramiv(program, GL_LINK_STATUS):
            raise RuntimeError(glGetProgramInfoLog(program))

        glDeleteShader(v)
        glDeleteShader(f)

        return program

    # ---------------------------
    # Node transform helpers
    # ---------------------------

    def _node_local_matrix(self, node):
        """Return the 4x4 local transform for a glTF node."""
        if node.matrix is not None:
            # glTF stores column-major; numpy is row-major
            return np.array(node.matrix, dtype=np.float32).reshape(4, 4).T

        s = node.scale or [1.0, 1.0, 1.0]
        S = np.diag([s[0], s[1], s[2], 1.0]).astype(np.float32)

        rx, ry, rz, rw = node.rotation if node.rotation else [0.0, 0.0, 0.0, 1.0]
        R = np.eye(4, dtype=np.float32)
        R[0, 0] = 1 - 2*(ry*ry + rz*rz); R[0, 1] = 2*(rx*ry - rz*rw); R[0, 2] = 2*(rx*rz + ry*rw)
        R[1, 0] = 2*(rx*ry + rz*rw);     R[1, 1] = 1 - 2*(rx*rx + rz*rz); R[1, 2] = 2*(ry*rz - rx*rw)
        R[2, 0] = 2*(rx*rz - ry*rw);     R[2, 1] = 2*(ry*rz + rx*rw);     R[2, 2] = 1 - 2*(rx*rx + ry*ry)

        t = node.translation or [0.0, 0.0, 0.0]
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = t[0]; T[1, 3] = t[1]; T[2, 3] = t[2]

        # glTF TRS order: translation * rotation * scale
        return T @ R @ S

    def _collect_mesh_transforms(self, gltf, node_idx, parent_xf, out):
        """Recursively walk the scene graph and map mesh_index → global 4x4 transform."""
        node = gltf.nodes[node_idx]
        global_xf = parent_xf @ self._node_local_matrix(node)

        if node.mesh is not None:
            out[node.mesh] = global_xf

        for child_idx in (node.children or []):
            self._collect_mesh_transforms(gltf, child_idx, global_xf, out)

    # ---------------------------
    # Load GLB
    # ---------------------------
    def load_glb(self, path):

        gltf = GLTF2().load(path)
        blob = gltf.binary_blob()

        mesh = PlyMesh()

        # Build mesh_index → global transform from the scene graph
        mesh_transforms = {}
        if gltf.scene is not None:
            scene = gltf.scenes[gltf.scene]
            for root_idx in (scene.nodes or []):
                self._collect_mesh_transforms(gltf, root_idx, np.eye(4, dtype=np.float32), mesh_transforms)

        vertices = []
        normals = []
        uvs = []
        indices = []

        vertex_offset = 0  # running offset

        for mesh_idx, m in enumerate(gltf.meshes):

            node_xf     = mesh_transforms.get(mesh_idx, np.eye(4, dtype=np.float32))
            normal_mat  = np.linalg.inv(node_xf[:3, :3]).T

            for prim in m.primitives:

                # -------------------------
                # POSITION
                # -------------------------
                pos_accessor = gltf.accessors[prim.attributes.POSITION]
                pos_view = gltf.bufferViews[pos_accessor.bufferView]

                pos_offset = (pos_view.byteOffset or 0) + (pos_accessor.byteOffset or 0)

                pos = np.frombuffer(
                    blob,
                    dtype=np.float32,
                    count=pos_accessor.count * 3,
                    offset=pos_offset
                ).reshape(-1, 3)

                # bake node transform into positions
                pos_h = np.hstack([pos, np.ones((len(pos), 1), dtype=np.float32)])
                pos   = (node_xf @ pos_h.T).T[:, :3]

                vertices.append(pos)

                # -------------------------
                # NORMAL
                # -------------------------
                if hasattr(prim.attributes, "NORMAL") and prim.attributes.NORMAL is not None:

                    norm_accessor = gltf.accessors[prim.attributes.NORMAL]
                    norm_view = gltf.bufferViews[norm_accessor.bufferView]

                    norm_offset = (norm_view.byteOffset or 0) + (norm_accessor.byteOffset or 0)

                    norm = np.frombuffer(
                        blob,
                        dtype=np.float32,
                        count=norm_accessor.count * 3,
                        offset=norm_offset
                    ).reshape(-1, 3)

                    # bake node transform into normals (use inverse-transpose of upper 3x3)
                    norm = (normal_mat @ norm.T).T
                    nlen = np.linalg.norm(norm, axis=1, keepdims=True)
                    nlen = np.where(nlen < 1e-8, 1.0, nlen)
                    norm = norm / nlen

                    normals.append(norm)

                else:
                    normals.append(np.zeros_like(pos))

                # -------------------------
                # UV
                # -------------------------
                if hasattr(prim.attributes, "TEXCOORD_0") and prim.attributes.TEXCOORD_0 is not None:

                    uv_accessor = gltf.accessors[prim.attributes.TEXCOORD_0]
                    uv_view = gltf.bufferViews[uv_accessor.bufferView]

                    uv_offset = (uv_view.byteOffset or 0) + (uv_accessor.byteOffset or 0)

                    uv = np.frombuffer(
                        blob,
                        dtype=np.float32,
                        count=uv_accessor.count * 2,
                        offset=uv_offset
                    ).reshape(-1, 2)

                    uvs.append(uv)

                else:
                    uvs.append(np.zeros((pos.shape[0], 2), dtype=np.float32))

                # -------------------------
                # INDICES
                # -------------------------
                idx_accessor = gltf.accessors[prim.indices]
                idx_view = gltf.bufferViews[idx_accessor.bufferView]

                idx_offset = (idx_view.byteOffset or 0) + (idx_accessor.byteOffset or 0)

                component = idx_accessor.componentType

                if component == 5121:
                    dtype = np.uint8
                elif component == 5123:
                    dtype = np.uint16
                elif component == 5125:
                    dtype = np.uint32
                else:
                    raise RuntimeError(f"Unsupported index type {component}")

                idx = np.frombuffer(
                    blob,
                    dtype=dtype,
                    count=idx_accessor.count,
                    offset=idx_offset
                ).astype(np.uint32)

                # shift indices for combined mesh
                idx = idx + vertex_offset

                indices.append(idx)

                vertex_offset += pos.shape[0]

                # -------------------------
                # MATERIAL  (first hit wins)
                # -------------------------
                if not mesh.has_texture and prim.material is not None:

                    mat = gltf.materials[prim.material]
                    pbr = mat.pbrMetallicRoughness

                    if pbr is not None:

                        if pbr.baseColorFactor is not None:
                            mesh.base_color = np.array(pbr.baseColorFactor, dtype=np.float32)

                        mesh.metallic_factor  = pbr.metallicFactor  if pbr.metallicFactor  is not None else 0.0
                        mesh.roughness_factor = pbr.roughnessFactor if pbr.roughnessFactor is not None else 1.0

                        if pbr.baseColorTexture is not None:

                            tex_index = pbr.baseColorTexture.index
                            tex       = gltf.textures[tex_index]
                            img       = gltf.images[tex.source]

                            view   = gltf.bufferViews[img.bufferView]
                            offset = view.byteOffset or 0
                            length = view.byteLength

                            img_bytes = blob[offset : offset + length]

                            pil      = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
                            tex_data = np.array(pil)

                            texture_id = glGenTextures(1)
                            glBindTexture(GL_TEXTURE_2D, texture_id)

                            glTexImage2D(
                                GL_TEXTURE_2D, 0, GL_RGBA,
                                tex_data.shape[1], tex_data.shape[0],
                                0, GL_RGBA, GL_UNSIGNED_BYTE, tex_data
                            )

                            glGenerateMipmap(GL_TEXTURE_2D)

                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

                            glBindTexture(GL_TEXTURE_2D, 0)

                            mesh.texture_id  = texture_id
                            mesh.has_texture = True

        # -----------------------------------
        # Combine buffers
        # -----------------------------------

        vertices = np.vstack(vertices)
        normals = np.vstack(normals)
        uvs = np.vstack(uvs)

        indices = np.concatenate(indices)

        mesh.vertex_count = len(vertices)
        mesh.index_count = len(indices)

        data = np.hstack([vertices, normals, uvs]).astype(np.float32)

        # -----------------------------------
        # Upload to OpenGL
        # -----------------------------------

        mesh.vao = glGenVertexArrays(1)
        mesh.vbo = glGenBuffers(1)
        mesh.ebo = glGenBuffers(1)

        glBindVertexArray(mesh.vao)

        glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        stride = data.strides[0]

        # position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, False, stride, ctypes.c_void_p(0))

        # normal
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, False, stride, ctypes.c_void_p(12))

        # uv
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, False, stride, ctypes.c_void_p(24))

        glBindVertexArray(0)

        mesh.type = 0

        return mesh

    # ---------------------------
    # Texture loader
    # ---------------------------

    def load_texture(self, path):

        img = Image.open(path).convert("RGBA")
        img = np.array(img)

        tex = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, tex)

        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            img.shape[1],
            img.shape[0],
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            img
        )

        glGenerateMipmap(GL_TEXTURE_2D)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glBindTexture(GL_TEXTURE_2D,0)

        return tex

    # ---------------------------
    # Render
    # ---------------------------

    def render_glb_mesh(self, mesh, shader):

        glUseProgram(shader)

        model = mesh.local_transform

        model_loc = glGetUniformLocation(shader,"model_matrix")
        view_loc = glGetUniformLocation(shader,"view_matrix")
        proj_loc = glGetUniformLocation(shader,"projection_matrix")

        glUniformMatrix4fv(model_loc,1,GL_FALSE,model)
        glUniformMatrix4fv(view_loc,1,GL_FALSE,self.view_matrix)
        glUniformMatrix4fv(proj_loc,1,GL_FALSE,self.proj_matrix)

        glUniform4fv(
            glGetUniformLocation(shader,"base_color"),
            1,
            mesh.base_color
        )

        glUniform1i(
            glGetUniformLocation(shader, "has_texture"),
            int(mesh.has_texture)
        )

        if mesh.has_texture:

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, mesh.texture_id)

            glUniform1i(
                glGetUniformLocation(shader, "base_color_texture"),
                0
            )

        glBindVertexArray(mesh.vao)

        glDrawElements(
            GL_TRIANGLES,
            mesh.index_count,
            GL_UNSIGNED_INT,
            None
        )

        glBindVertexArray(0)