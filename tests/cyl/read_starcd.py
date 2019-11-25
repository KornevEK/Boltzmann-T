import numpy as np
class Mesh:
    #
    # Computes volume of tetrahedron from coord-s of vertices
    #
    def compute_tetra_volume(self, tetra):
        A = np.zeros((3,3))
        A[:, 0] = tetra[:, 1] - tetra[:, 0]
        A[:, 1] = tetra[:, 2] - tetra[:, 0]
        A[:, 2] = tetra[:, 3] - tetra[:, 0]
        return np.linalg.det(A) / 6.

    def read_starcd(self, path, scale = 1):
        max_vert_in_face = 4
        max_vert_in_cell = 8
        #
        # Read vertex list and bc type for each boundary face 
        #
        data = np.loadtxt(path + 'star.bnd', usecols=(1,2,3,4,5))
        self.nbf = data.shape[0]
        print 'Number of boundary faces = ', self.nbf
        self.bcface_vert_lists = data[:,0:-1].astype(np.int) - 1 # -1 due to Python counts from 0
        self.bcface_bctype = data[:,-1].astype(np.int) - 1
        #
        # Construct list of boundary faces indices for each bctype
        #
        self.nbc = len(set(self.bcface_bctype)) # Number of different boundary conditions
        print 'Number of boundary conditions = ', self.nbc
        self.bf_for_each_bc = []
        for i in range(self.nbc):
            self.bf_for_each_bc.append(np.argwhere(self.bcface_bctype == i)[:,0])
        #
        # Count number of cells
        #
        file = open(path + 'star.cel')
        nc = 0
        for i, l in enumerate(file):
            verts = np.array(list(map(int, l.split()[1:max_vert_in_cell+1])), dtype = np.int)
            # Check, is it "shell" cell?
            if(np.all(verts[4::] != 0)):
                nc += 1 # else it is shell   
        file.close() 
        self.nc = nc
        print 'Number of cells = ', self.nc
        # Count number of vertices
        file = open(path + 'star.vrt')
        for i, l in enumerate(file):
            pass 
        file.close() 
        self.nv = i + 1
        print 'Number of vertices = ', self.nv

        # Allocate arrays
        self.bc_index_list = np.zeros(shape = (self.nbf), dtype = np.int)
        # for each face - list of vertices' indices
        self.bc_face_vertices = np.zeros(shape = (self.nbf, max_vert_in_face), dtype = np.int)

        self.vert_coo = np.zeros((self.nv, 3))
        self.vert_list_for_cell = np.zeros((self.nc, max_vert_in_cell), dtype = int)
        # Read bc index for each boundary face
        file = open(path + 'star.bnd')
        for i, l in enumerate(file):
            tokens = l.split()
            for j in range(max_vert_in_face):
                self.bc_face_vertices[i, j] = int(tokens[j+1])
            self.bc_index_list[i] = int(tokens[j+2]) # index of boundary condition
        file.close()
        # Compute number of faces in each boundary condition
        pass
        #
        # Read vertices's coordinates
        #
        self.vert_coo = scale * np.loadtxt(fname = path + 'star.vrt', usecols=(1,2,3))
        #
        # Read vertex lists for cells
        #
        file = open(path + 'star.cel')
        j = 0
        for i, l in enumerate(file):
            verts = np.array(list(map(int, l.split()[1:max_vert_in_cell+1])), dtype = np.int)
            # Check, is it "shell" cell?
            if np.all(verts[4::] != 0): # else it is shell
                # Convert order to StarCD
                verts = verts[[4, 5, 7, 6, 0, 1, 3, 2]]
                # Convert order to Gambit
                verts = verts[[6, 7, 2, 3, 4, 5, 0, 1]]
                self.vert_list_for_cell[j,:] = verts - 1 # since Python counts from 0
                j += 1   
        file.close()
        #
        # Calculate cell centers - arithmetic mean of vertises' coordinates
        #
        self.cell_center_coo = np.zeros((nc, 3))
        for i in range(nc):
            verts_inds = self.vert_list_for_cell[i,:]
            self.cell_center_coo[i,:] = np.sum(self.vert_coo[verts_inds,:],axis=0)/max_vert_in_cell
        #
        # Calculate volume of each cell
        #
        faces = np.zeros((4,6), dtype = np.int) # 4 verices in each of 6 faces
        tetra = np.zeros((3, 4)) # 3 - x, y, z coordinates; 4 - number of vertex in tetra
        self.cell_volumes = np.zeros(nc)
        for ic in range(self.nc):
            verts = self.vert_list_for_cell[ic,:]
            # construct faces of cell
            faces[:, 0] = verts[[0, 1, 5, 4]]
            faces[:, 1] = verts[[1, 3, 7, 5]]
            faces[:, 2] = verts[[3, 2, 6, 7]]
            faces[:, 3] = verts[[2, 0, 4, 6]]
            faces[:, 4] = verts[[1, 0, 2, 3]]
            faces[:, 5] = verts[[4, 5, 7, 6]]
            # Loop over faces, for each face construct 4 tetras 
            # and compute their volumes
            for jf in range(6):
                face_center = np.sum(self.vert_coo[faces[:, jf], :], axis = 0)/4
                x1 = self.vert_coo[faces[0, jf], :]
                x2 = self.vert_coo[faces[1, jf], :]
                x3 = self.vert_coo[faces[2, jf], :]
                x4 = self.vert_coo[faces[3, jf], :]
                # 1st tetra
                tetra[:, 0] = self.cell_center_coo[ic, :]
                tetra[:, 1] = x1
                tetra[:, 2] = x2 
                tetra[:, 3] = face_center
                self.cell_volumes[ic] += self.compute_tetra_volume(tetra)
                # 2nd tetra
                tetra[:, 0] = self.cell_center_coo[ic, :]
                tetra[:, 1] = x2
                tetra[:, 2] = x3
                tetra[:, 3] = face_center
                self.cell_volumes[ic] += self.compute_tetra_volume(tetra)
                # 3rd tetra
                tetra[:, 0] = self.cell_center_coo[ic, :]
                tetra[:, 1] = x3
                tetra[:, 2] = x4
                tetra[:, 3] = face_center
                self.cell_volumes[ic] += self.compute_tetra_volume(tetra)
                # 4th tetra
                tetra[:, 0] = self.cell_center_coo[ic, :]
                tetra[:, 1] = x4
                tetra[:, 2] = x1
                tetra[:, 3] = face_center
                self.cell_volumes[ic] += self.compute_tetra_volume(tetra)
        #
        # Construct for each vertex list of cells to which it belongs
        #
        self.cell_list_for_vertex = -np.ones((self.nv, 8), dtype = np.int) # it may be > 8!
        self.cell_num_for_vertex = np.zeros(self.nv, dtype = np.int) # Number of cells, adjacent to each vertex
        for ic in range(self.nc):
            for jv in range(max_vert_in_cell):
                vert = self.vert_list_for_cell[ic, jv] # global vertex index
                self.cell_list_for_vertex[vert, self.cell_num_for_vertex[vert]] = ic
                self.cell_num_for_vertex[vert] += 1

        #
        # Construct for each cell list of neighboring cells 
        #
        self.cell_neighbors_list = -np.ones((self.nc,6), dtype = np.int) # -1 means no neighbor
        faces_neigh = np.zeros((4,6), dtype = np.int)
        self.face_vert_list = np.zeros((6 * self.nc, 4), dtype = np.int)
        self.cell_face_list = -np.ones((self.nc, 6), dtype = np.int)
        nf = 0 # Number of faces
        for ic in range(self.nc):
            verts = self.vert_list_for_cell[ic,:]
            # construct faces of cell
            faces[:, 0] = verts[[0, 1, 5, 4]]
            faces[:, 1] = verts[[1, 3, 7, 5]]
            faces[:, 2] = verts[[3, 2, 6, 7]]
            faces[:, 3] = verts[[2, 0, 4, 6]]
            faces[:, 4] = verts[[1, 0, 2, 3]]
            faces[:, 5] = verts[[4, 5, 7, 6]]
            for jf in range(6): # loop over faces
                if (self.cell_neighbors_list[ic,jf] >= 0): # if face is already assigned - skip
                    continue
                self.face_vert_list[nf, :] = faces[:, jf] # add face to global list
                self.cell_face_list[ic, jf] = nf
                # loop over all vertices in face
                for iv in range(4):
                    # loop over all cells containing this vertex
                    for kc in range(self.cell_num_for_vertex[faces[iv, jf]]): 
                        icell = self.cell_list_for_vertex[faces[iv, jf], kc]
                        verts_neigh = self.vert_list_for_cell[icell, :]
                        # construct faces of cell
                        faces_neigh[:, 0] = verts_neigh[[0, 1, 5, 4]]
                        faces_neigh[:, 1] = verts_neigh[[1, 3, 7, 5]]
                        faces_neigh[:, 2] = verts_neigh[[3, 2, 6, 7]]
                        faces_neigh[:, 3] = verts_neigh[[2, 0, 4, 6]]
                        faces_neigh[:, 4] = verts_neigh[[1, 0, 2, 3]]
                        faces_neigh[:, 5] = verts_neigh[[4, 5, 7, 6]]
                        # Now compare these faces with fase 
                        for lf in range(6):
                            if np.all(np.sort(faces[:, jf]) == np.sort(faces_neigh[:, lf])):
                                self.cell_face_list[icell, lf] = nf
                                self.cell_neighbors_list[ic, jf] = icell
                                self.cell_neighbors_list[icell, lf] = ic
                nf += 1
        self.nf = nf
        print 'Number of faces = ', self.nf
        self.face_vert_list = self.face_vert_list[:self.nf, :] # exlude etra rows
        #
        print 'sum of volumes:', np.sum(self.cell_volumes)
        #
        # Compute face areas and normals
        #
        self.face_areas = np.zeros(nf)
        self.face_normals = np.zeros((nf,3))
        for jf in range(nf):
            verts = self.face_vert_list[jf,:]
            verts_coo = self.vert_coo[verts, :]
            v5 = np.sum(verts_coo, axis = 0) # face center
            
            vec1 = 0.5*(verts_coo[2,:] + verts_coo[1,:]) - 0.5*(verts_coo[0,:] + verts_coo[3,:]) 
            vec2 = 0.5*(verts_coo[3,:] + verts_coo[2,:]) - 0.5*(verts_coo[1,:] + verts_coo[0,:])
            self.face_areas[jf] = np.linalg.norm(np.cross(vec1, vec2), 2)
            #
            # Complicated procedure to overcome problems when area is tiny
            #
            len1 = np.linalg.norm(vec1, 2)
            len2 = np.linalg.norm(vec2, 2)
            bec1 = vec1  / np.maximum(1e-13,len1)
            bec2 = vec2  / np.maximum(1e-13,len2)
            normal =  np.cross(bec1,bec2)
            length = np.linalg.norm(normal, 2)        
            if length <= 1e-10: 
                flag = False
            else:    
                normal = normal/length
                flag = True
            self.face_normals[jf,:] = normal
        
        #
        # Compute orientation of face normals with respect to each cell
        #
        # +1 - outer normal, -1 - inner normal (directed in cell)
        self.cell_face_normal_direction = np.zeros((self.nc, 6), dtype = np.int) 
        for ic in range(self.nc):
            for jf in range(6):
                face = self.cell_face_list[ic, jf]
                face_normal = self.face_normals[face, :]
                face_verts = self.face_vert_list[face,:]
                face_center = np.sum(self.vert_coo[face_verts, :], axis = 0) / 4
                # Compute vector from cell center to center of face
                vec = face_center - self.cell_center_coo[ic, :]
                dot_prod = np.dot(vec, face_normal)
                if dot_prod >= 0:
                    self.cell_face_normal_direction[ic, jf] = +1
                else:
                    self.cell_face_normal_direction[ic, jf] = -1
                    
        self.bound_face_info = np.zeros((self.nbf, 3), dtype = np.int)
        for ibf in range(self.nbf):
            for jf in range(self.nf):
                if (set(self.bcface_vert_lists[ibf, :]) == set(self.face_vert_list[jf, :])):
                    self.bound_face_info[ibf, 0] = jf
            
            for ic in range(self.nc):
                for jf in range(6):
                    if (self.cell_face_list[ic, jf] == self.bound_face_info[ibf, 0]):
                        self.bound_face_info[ibf, 2] = self.cell_face_normal_direction[ic, jf]
                    
            self.bound_face_info[ibf, 1] = self.bcface_bctype[ibf]
            
        self.cell_diam = np.zeros(self.nc)
        face_diam = np.zeros(6)
        for ic in range(self.nc):
            for jf in range(6):
                face = self.cell_face_list[ic, jf]
                face_verts = self.face_vert_list[face,:]
                face_center = np.sum(self.vert_coo[face_verts, :], axis = 0) / 4
                vec = face_center - self.cell_center_coo[ic, :]
                face_diam[jf] = 2 * np.linalg.norm(vec)
            self.cell_diam[ic] = np.min(face_diam)
            
def write_tecplot(mesh, data, fname, var_names, time = 0.0):
    '''
    Procedure writes solution in cell centers of an unstructured mesh in Tecplot ASCII format
    '''
    nv = data.shape[1] # number of variables
    file = open(fname, mode = 'w')
    file.write('TITLE = "VolumeData"\n')
    file.write('VARIABLES = "x" "y" "z" ')
    for iv in range(nv):
        file.write(' "' + var_names[iv] + '" ')
    file.write('\n')
    file.write('ZONE T= my_zone, SolutionTime = ' + str(time) + 
               ', DATAPACKING=Block, ZONETYPE=FEBRICK Nodes= ' + str(mesh.nv) +
              ' Elements= ' + str(mesh.nc))
    file.write(' VarLocation=([4-'+ str(3+nv) + ']=CellCentered)')
    # write vertices' coo
    for i in range(3):
        for iv in range(mesh.nv):
            file.write('{:20.10e}'.format(mesh.vert_coo[iv,i]) +'\n')
    # Write values of variables
    for i in range(nv):
        for ic in range(mesh.nc):
            file.write('{:20.10e}'.format(data[ic, i]) + '\n')
    # Write cell-to-vertices connectivity
    for ic in range(mesh.nc):
        verts = mesh.vert_list_for_cell[ic,:]
        # Reorder verts for Tecplot
        # tecplot numbering corresponds to gambit numbeting
        # 4 5 1 0
        # 6 7 3 2
        verts = verts[[4, 5, 1, 0, 6, 7, 3, 2]]
        for j in range(8):
            file.write('{0:d}'.format(verts[j] + 1) + ' ')
        file.write('\n')