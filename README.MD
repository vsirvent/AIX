# Project Build Instructions

Follow the following steps to build the project and make it deployment ready.

---

### Step 1: Install Dependencies

For macOS users, you have two options to install the required dependencies. Please choose one of the following methods:

#### Using MacPorts:
```bash
sudo port install libomp
```

#### Using Brew:
```bash
brew install libomp
```

### Step 2: Build Externals

This step will build external libraries and download MNIST data set.

```bash
cd Externals
./build_all.sh
```

### Step 3: Build Targets

This step will build all binaries and deploy into a specific folder.

```bash
./build.sh release product-rel
```

After the successful build, all target binaries will be deployed into the product-rel folder.

Note: Run the build.sh file without parameters to see all options.
