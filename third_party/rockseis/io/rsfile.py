import numpy as np
import os
import struct as st
import tempfile


MAGICNUMBER = "R0CKS"  # rss identifier
MAGICNUMBERLENGTH = 5
MAXDIMS = 9


class rsfile:
    def __init__(self, data=None, datatype=None):
        ndim = 0
        dims = []
        if data is not None:
            ndim = data.ndim
            dims = data.shape
            self.data = data
            if datatype is not None:
                if datatype < 2 or datatype > 3:
                    raise TypeError("Invalid datatype: use 2 for DATA2D and 3 for DATA3D formats")
                if ndim > 2:
                    raise TypeError("For DATA2D and DATA3D formats ndim = 2 is expeted")
        self.data_format = 4
        self.header_format = 4
        self.Nheader = 0
        self.type = 0
        if datatype == 2:
            self.type = 2
            self.Nheader = 4
        if datatype == 3:
            self.type = 3
            self.Nheader = 6
        self.Ndims = ndim
        self.geomN = np.zeros([MAXDIMS], dtype="uint64")
        self.geomD = np.zeros([MAXDIMS], dtype="float64")
        self.geomO = np.zeros([MAXDIMS], dtype="float64")
        self.fullsize = 1
        for i in range(0, ndim):
            self.geomN[i] = np.uint64(dims[i])
            self.geomD[i] = 1.0
            self.fullsize = self.fullsize * self.geomN[i]
        self.fullsize = int(self.fullsize)

        self.srcX = np.zeros([int(self.geomN[1])], dtype="float32")
        self.srcZ = np.zeros([int(self.geomN[1])], dtype="float32")
        self.GroupX = np.zeros([int(self.geomN[1])], dtype="float32")
        self.GroupZ = np.zeros([int(self.geomN[1])], dtype="float32")
        if self.Nheader == 6:
            self.srcY = np.zeros([int(self.geomN[1])], dtype="float32")
            self.GroupY = np.zeros([int(self.geomN[1])], dtype="float32")

    def read(self, filename):
        with open(filename, "rb") as f:
            if f.read(MAGICNUMBERLENGTH).decode("utf-8") != MAGICNUMBER:
                print("Not a valid rss file!")
                return -1

            self.data_format = st.unpack("i", f.read(4))[0]
            self.header_format = st.unpack("i", f.read(4))[0]
            self.type = st.unpack("i", f.read(4))[0]
            self.Nheader = st.unpack("Q", f.read(8))[0]
            self.Ndims = st.unpack("Q", f.read(8))[0]

            self.geomN = np.zeros(MAXDIMS, dtype="uint64")
            self.geomD = np.zeros(MAXDIMS, dtype="float64")
            self.geomO = np.zeros(MAXDIMS, dtype="float64")

            for i in range(MAXDIMS):
                self.geomN[i] = st.unpack("Q", f.read(8))[0]
            for i in range(MAXDIMS):
                self.geomD[i] = st.unpack("d", f.read(8))[0]
            for i in range(MAXDIMS):
                self.geomO[i] = st.unpack("d", f.read(8))[0]

            valid_dims = self.geomN > 0
            self.Ndims = np.sum(valid_dims)
            self.fullsize = int(np.prod(self.geomN[valid_dims]))
            resize = [int(n) for n in self.geomN[valid_dims]]

            dtype = "float32" if self.data_format == 4 else "float64"
            self.data = np.zeros(self.fullsize, dtype=dtype)

            if self.Nheader:
                n_coords = int(self.geomN[1])
                coord_arrays = {
                    "srcX": np.zeros(n_coords, dtype="float32"),
                    "srcZ": np.zeros(n_coords, dtype="float32"),
                    "GroupX": np.zeros(n_coords, dtype="float32"),
                    "GroupZ": np.zeros(n_coords, dtype="float32"),
                }

                if self.Nheader == 6:
                    coord_arrays.update(
                        {
                            "srcY": np.zeros(n_coords, dtype="float32"),
                            "GroupY": np.zeros(n_coords, dtype="float32"),
                        }
                    )

                chunk_size = int(self.geomN[0])
                for i in range(n_coords):
                    if self.Nheader == 4:
                        coords = st.unpack("4f", f.read(4 * self.header_format))
                        (
                            coord_arrays["srcX"][i],
                            coord_arrays["srcZ"][i],
                            coord_arrays["GroupX"][i],
                            coord_arrays["GroupZ"][i],
                        ) = coords
                    else:
                        coords = st.unpack("6f", f.read(6 * self.header_format))
                        (
                            coord_arrays["srcX"][i],
                            coord_arrays["srcY"][i],
                            coord_arrays["srcZ"][i],
                            coord_arrays["GroupX"][i],
                            coord_arrays["GroupY"][i],
                            coord_arrays["GroupZ"][i],
                        ) = coords

                    start_idx = i * chunk_size
                    end_idx = (i + 1) * chunk_size
                    self.data[start_idx:end_idx] = np.fromfile(f, dtype=dtype, count=chunk_size)

                for name, array in coord_arrays.items():
                    setattr(self, name, array)
            else:
                self.data = np.fromfile(f, dtype=dtype, count=self.fullsize)

            self.data = np.reshape(self.data, resize, order="F")

    def readheader(self, filename):
        f = open(filename, "rb")
        buff = f.read(MAGICNUMBERLENGTH)

        if buff.decode("utf-8") != MAGICNUMBER:
            print("Not a valid rss file!")
            return -1

        self.data_format = st.unpack("i", f.read(4))[0]
        self.header_format = st.unpack("i", f.read(4))[0]
        self.type = st.unpack("i", f.read(4))[0]
        self.Nheader = st.unpack("Q", f.read(8))[0]
        self.Ndims = st.unpack("Q", f.read(8))[0]
        for i in range(0, MAXDIMS):
            self.geomN[i] = st.unpack("Q", f.read(8))[0]
        for i in range(0, MAXDIMS):
            self.geomD[i] = st.unpack("d", f.read(8))[0]
        for i in range(0, MAXDIMS):
            self.geomO[i] = st.unpack("d", f.read(8))[0]

        self.fullsize = 1
        self.Ndims = 0
        for i in range(0, MAXDIMS):
            if self.geomN[i] > 0:
                self.fullsize = self.fullsize * self.geomN[i]
                self.Ndims = self.Ndims + 1

        self.fullsize = int(self.fullsize)

    def window(self, filename, par):
        f = tempfile.NamedTemporaryFile(delete=False)
        os.system("rswindow " + par + " < " + filename + " > " + f.name)
        self.read(f.name)
        os.unlink(f.name)

    def info(self, filename):
        self.readheader(filename)
        for i in range(0, MAXDIMS):
            if self.geomN[i]:
                print(
                    "N"
                    + str(i + 1)
                    + "="
                    + str(self.geomN[i])
                    + "    "
                    + "D"
                    + str(i + 1)
                    + "="
                    + str(self.geomD[i])
                    + "    "
                    + "O"
                    + str(i + 1)
                    + "="
                    + str(self.geomO[i])
                )

    def write(self, filename):
        with open(filename, "wb") as f:
            f.write(MAGICNUMBER.encode())
            f.write(st.pack("i", self.data_format))
            f.write(st.pack("i", self.header_format))
            f.write(st.pack("i", self.type))
            f.write(st.pack("Q", self.Nheader))
            f.write(st.pack("Q", self.Ndims))

            for i in range(MAXDIMS):
                f.write(st.pack("Q", self.geomN[i]))
            for i in range(MAXDIMS):
                f.write(st.pack("d", self.geomD[i]))
            for i in range(MAXDIMS):
                f.write(st.pack("d", self.geomO[i]))

            valid_dims = self.geomN > 0
            self.Ndims = np.sum(valid_dims)
            self.fullsize = int(np.prod(self.geomN[valid_dims]))
            dtype = "float32" if self.data_format == 4 else "float64"

            if self.Nheader:
                n_coords = int(self.geomN[1])
                coord_fmt = "4f" if self.Nheader == 4 else "6f"
                dataout = self.data.astype(dtype)
                for i in range(n_coords):
                    if self.Nheader == 4:
                        f.write(st.pack(coord_fmt, self.srcX[i], self.srcZ[i], self.GroupX[i], self.GroupZ[i]))
                    else:
                        f.write(
                            st.pack(
                                coord_fmt,
                                self.srcX[i],
                                self.srcY[i],
                                self.srcZ[i],
                                self.GroupX[i],
                                self.GroupY[i],
                                self.GroupZ[i],
                            )
                        )
                    dataout[:, i].tofile(f)
            else:
                dataout = self.data.reshape(self.fullsize, order="F").astype(dtype)
                dataout.tofile(f)
