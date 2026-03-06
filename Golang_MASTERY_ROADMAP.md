# GOLANG MASTERY ROADMAP: From Zero to Staff-Level Infrastructure Engineer

> **Scope:** Beginner to Expert | DevOps, Platform Engineering, Kubernetes, MLOps, LLMOps, Distributed Systems
>
> **Format:** Checklist-based, GitHub-ready, Interview-focused
>
> **Progression:** Absolute Beginner -> Intermediate -> Advanced -> Production Systems -> Staff-Level Distributed Systems

---

## Table of Contents

| Stage | Title | Level |
|-------|-------|-------|
| 1 | Core Foundations: Linux, CLI, Environment Setup | Absolute Beginner |
| 2 | Go Language Fundamentals | Beginner |
| 3 | Go Idioms and Best Practices | Beginner-Intermediate |
| 4 | Object-Oriented Design in Go | Intermediate |
| 5 | Advanced Go: Concurrency and the Runtime | Intermediate-Advanced |
| 6 | Go for Systems Programming | Advanced |
| 7 | Go Networking Deep Dive | Advanced |
| 8 | API Development (Production-Grade) | Advanced |
| 9 | Databases and Data Persistence | Advanced |
| 10 | Testing, Benchmarking, and Profiling | Advanced |
| 11 | Go for DevOps: CLI Tools and Automation | Advanced |
| 12 | Security in Go | Advanced |
| 13 | Containers and Kubernetes Fundamentals | Advanced |
| 14 | Kubernetes Controllers and Operators | Expert |
| 15 | Observability and Instrumentation | Expert |
| 16 | Distributed Systems | Expert |
| 17 | MLOps Infrastructure | Expert |
| 18 | LLMOps Infrastructure | Expert |
| 19 | GPU Infrastructure and ML Platforms | Expert |
| 20 | System Design for Infrastructure Interviews | Expert |
| 21 | Capstone Projects | Staff-Level |

---

---

# STAGE 1: CORE FOUNDATIONS — Linux, CLI, Environment Setup

---

## 1.1 Linux Fundamentals for Go Engineers

### Concepts to Learn

- [ ] Linux kernel architecture overview (kernel space vs user space)
- [ ] Linux distributions commonly used in production (Ubuntu, CentOS/RHEL, Alpine, Debian)
- [ ] The Linux filesystem hierarchy (`/`, `/etc`, `/var`, `/usr`, `/tmp`, `/proc`, `/sys`, `/dev`)
- [ ] File permissions model: owner, group, other; read, write, execute
- [ ] Numeric permissions (chmod 755, 644, 600)
- [ ] Special permissions: setuid, setgid, sticky bit
- [ ] Users and groups: `useradd`, `groupadd`, `usermod`, `id`, `whoami`
- [ ] Process model: PID, PPID, process states (running, sleeping, zombie, stopped)
- [ ] Process management: `ps`, `top`, `htop`, `kill`, `killall`, `nice`, `renice`
- [ ] Signals: SIGTERM, SIGKILL, SIGINT, SIGHUP, SIGUSR1, SIGUSR2
- [ ] Systemd: units, services, timers, `systemctl`, `journalctl`
- [ ] Package management: `apt`, `yum`, `dnf`, `apk`
- [ ] Environment variables: `export`, `env`, `printenv`, `.bashrc`, `.profile`
- [ ] PATH variable and how binary resolution works
- [ ] Filesystem operations: `ls`, `cd`, `pwd`, `mkdir`, `rmdir`, `rm`, `cp`, `mv`, `ln`
- [ ] Symbolic links vs hard links
- [ ] File inspection: `cat`, `less`, `more`, `head`, `tail`, `wc`, `file`, `stat`
- [ ] Text processing: `grep`, `sed`, `awk`, `cut`, `sort`, `uniq`, `tr`, `xargs`
- [ ] Pipe operator `|` and I/O redirection (`>`, `>>`, `<`, `2>`, `2>&1`, `&>`)
- [ ] Disk and storage: `df`, `du`, `mount`, `umount`, `fdisk`, `lsblk`
- [ ] Networking basics: `ip`, `ifconfig`, `netstat`, `ss`, `ping`, `traceroute`, `dig`, `nslookup`, `curl`, `wget`
- [ ] Firewall basics: `iptables`, `ufw`, `firewalld`
- [ ] SSH: key generation, `ssh-keygen`, `ssh-copy-id`, `ssh-agent`, config file, tunneling
- [ ] `/proc` filesystem: reading CPU info, memory info, process details
- [ ] `/sys` filesystem and device interaction
- [ ] cgroups and namespaces (foundational for containers)
- [ ] `strace` and `ltrace` for system call tracing
- [ ] `lsof` for open file descriptors
- [ ] `dmesg` for kernel ring buffer messages

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Every DevOps engineer lives in Linux. You debug production servers, manage services with systemd, trace issues with strace, manage firewall rules, and automate with shell scripts. |
| **Platform Engineering** | Platform engineers build tooling that runs on Linux. Understanding the filesystem hierarchy, cgroups, and namespaces is critical for container platforms. |
| **Kubernetes** | Kubernetes nodes are Linux. Pod networking uses Linux network namespaces, iptables/nftables. Debugging node issues requires deep Linux knowledge. |
| **MLOps** | ML training jobs run on Linux with NVIDIA GPU drivers. You need to understand CUDA driver loading, device files in `/dev`, and GPU monitoring. |
| **LLM Infrastructure** | LLM serving requires understanding memory management, huge pages, NUMA topology, and process affinity on Linux. |
| **Cloud Infrastructure** | Cloud VMs are Linux. You manage fleets of Linux servers, automate provisioning, and debug networking issues. |
| **GPU ML Systems** | GPU access happens through Linux device drivers. Understanding `/dev/nvidia*`, `nvidia-smi`, and kernel module management is essential. |
| **Distributed Systems** | Distributed systems rely on Linux networking (TCP tuning, socket options), process management, and signal handling for graceful shutdowns. |

### Practical Exercises

- [ ] Set up an Ubuntu VM (use VirtualBox, Multipass, or WSL2)
- [ ] Navigate the filesystem and identify the purpose of each top-level directory
- [ ] Create a user, assign groups, and set file permissions on a shared directory
- [ ] Run a long-running process, send it to background (`&`, `bg`, `fg`, `nohup`), and manage it
- [ ] Write a pipeline that finds all `.go` files, counts lines in each, and sorts by line count
- [ ] Set up SSH key-based authentication to a remote server or VM
- [ ] Use `strace` to trace system calls of a running process
- [ ] Explore `/proc/cpuinfo`, `/proc/meminfo`, `/proc/<pid>/status`
- [ ] Create a systemd service that runs a simple script on boot
- [ ] Use `ss -tlnp` to see listening ports and identify which process owns them

### Mini Projects

- [ ] **System Health Reporter**: Write a bash script that collects CPU, memory, disk, and network stats and outputs a formatted report
- [ ] **Log Analyzer**: Parse `/var/log/syslog` or `/var/log/auth.log` to extract failed SSH attempts, top source IPs, and time patterns
- [ ] **Process Monitor**: Script that monitors a given PID and alerts (prints/logs) when memory usage exceeds a threshold

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Reverse a String | LeetCode #344 | Easy | Basic string/array manipulation, warm-up |
| FizzBuzz | HackerRank | Easy | Basic control flow and conditionals |
| Two Sum | LeetCode #1 | Easy | Hash maps, foundational problem-solving |
| Valid Parentheses | LeetCode #20 | Easy | Stack basics, useful for parsing |
| Counting Sort (sort characters) | HackerRank | Easy | Understanding sorting and counting |

### Interview Focused Notes

**Common Interview Questions:**
- What happens when you type `ls -la` and press Enter? (Full path: shell lookup, fork, exec, syscalls, kernel, output to stdout)
- Explain file permissions 755 vs 644
- What is the difference between a process and a thread?
- How does SSH key authentication work?
- What are Linux namespaces and cgroups?
- How would you debug a process consuming too much memory on a production server?

**Common Mistakes:**
- Running commands as root when not necessary
- Not understanding the difference between `SIGTERM` and `SIGKILL` (graceful vs forced)
- Confusing hard links and symbolic links
- Not knowing that `rm` is irreversible (no trash)

**Interviewer Expectations:**
- At junior level: comfortable navigating Linux, basic commands, file permissions
- At mid level: process management, systemd, networking commands, shell pipelines
- At senior level: cgroups, namespaces, strace debugging, kernel parameters, performance tuning

---

## 1.2 Shell Scripting Basics

### Concepts to Learn

- [ ] Bash script structure: shebang (`#!/bin/bash`), comments, execution
- [ ] Variables: declaration, assignment, referencing (`$VAR`, `${VAR}`)
- [ ] String operations: concatenation, substring, length
- [ ] Arithmetic: `$(( ))`, `expr`, `let`
- [ ] Conditionals: `if`, `elif`, `else`, `fi`; test expressions `[ ]` and `[[ ]]`
- [ ] Comparison operators: `-eq`, `-ne`, `-gt`, `-lt`, `-ge`, `-le`, `==`, `!=`
- [ ] File test operators: `-f`, `-d`, `-e`, `-r`, `-w`, `-x`, `-s`
- [ ] Loops: `for`, `while`, `until`, `select`
- [ ] Arrays: indexed arrays, associative arrays
- [ ] Functions: declaration, arguments (`$1`, `$2`, `$@`, `$#`), return values
- [ ] Exit codes: `$?`, `exit 0`, `exit 1`
- [ ] Trap: catching signals in scripts (`trap 'cleanup' EXIT SIGTERM SIGINT`)
- [ ] Here documents and here strings (`<<EOF`, `<<<`)
- [ ] Command substitution: `$(command)` and backticks
- [ ] `set -e`, `set -u`, `set -o pipefail`, `set -x` for robust scripts
- [ ] `getopts` for argument parsing
- [ ] `cron` and `crontab` for scheduled tasks
- [ ] Script debugging: `bash -x script.sh`

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Shell scripts are the glue of CI/CD pipelines. You write deployment scripts, health checks, log rotators, and automation hooks. |
| **Platform Engineering** | Init containers in Kubernetes often use shell scripts. Entrypoint scripts configure runtime environments. |
| **Kubernetes** | Helm chart hooks, init containers, readiness/liveness probes sometimes invoke shell commands. |
| **MLOps** | Training job orchestration scripts, data preprocessing pipelines, and environment setup scripts are often in bash. |
| **LLM Infrastructure** | Model download scripts, checkpoint management, environment variable configuration for serving frameworks. |
| **Cloud Infrastructure** | Cloud-init scripts, VM provisioning, bootstrap scripts for fleet management. |
| **GPU ML Systems** | NVIDIA driver installation scripts, CUDA environment setup, GPU health check scripts. |
| **Distributed Systems** | Cluster bootstrap scripts, node initialization, service discovery registration scripts. |

### Practical Exercises

- [ ] Write a script that takes a directory path and lists all files sorted by size
- [ ] Write a script that monitors a log file in real-time and alerts on specific patterns
- [ ] Write a deployment script that pulls code from git, builds, and restarts a service
- [ ] Write a script that backs up a PostgreSQL database to S3 using `pg_dump` and `aws s3 cp`
- [ ] Write a script that takes CLI arguments (`--name`, `--env`, `--dry-run`) using `getopts`
- [ ] Set up a cron job that runs a cleanup script every day at 2 AM

### Mini Projects

- [ ] **Server Provisioning Script**: Automate the setup of a dev server (install packages, create users, configure SSH, set up firewall)
- [ ] **CI Pipeline Script**: Shell script that runs tests, builds a binary, packages it into a Docker image, and pushes to a registry
- [ ] **Multi-Environment Deployer**: Script that reads a config file and deploys to dev/staging/prod based on arguments

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Running Sum of 1D Array | LeetCode #1480 | Easy | Array iteration fundamentals |
| Maximum Subarray | LeetCode #53 | Medium | Kadane's algorithm, thinking about optimality |
| Contains Duplicate | LeetCode #217 | Easy | Hash set usage |
| Move Zeroes | LeetCode #283 | Easy | Two-pointer basics |
| Merge Sorted Array | LeetCode #88 | Easy | Array merge logic |

### Interview Focused Notes

**Common Interview Questions:**
- What does `set -euo pipefail` do and why should every production script use it?
- How do you pass arguments to a bash function?
- What is the difference between `$@` and `$*`?
- How would you write a script that gracefully handles SIGTERM?
- What is a race condition in shell scripts and how do you prevent it (lock files)?

**Common Mistakes:**
- Not quoting variables (`$VAR` vs `"$VAR"`) causing word splitting
- Forgetting `set -e` and ignoring failing commands
- Using `[ ]` instead of `[[ ]]` (latter handles empty strings better)
- Not handling cleanup on script exit (missing `trap`)

**Interviewer Expectations:**
- Can write production-quality scripts with error handling
- Understands signal handling and graceful cleanup
- Knows how to parse arguments and validate input
- Writes idempotent scripts (safe to run multiple times)

---

## 1.3 Development Environment Setup

### Concepts to Learn

- [ ] Go installation methods: official binary, package manager, `goenv`
- [ ] Go version management: installing multiple versions, switching between them
- [ ] Understanding `GOROOT` (where Go is installed)
- [ ] Understanding `GOPATH` (Go workspace: `src/`, `pkg/`, `bin/`)
- [ ] Go workspace layout (pre-modules legacy) vs modern module mode
- [ ] `GOPATH/bin` and adding it to `PATH`
- [ ] Go modules: `go.mod`, `go.sum`
- [ ] `go mod init`, `go mod tidy`, `go mod vendor`, `go mod download`
- [ ] Module versioning: semantic versioning, module paths, major version suffixes
- [ ] Go proxy: `GOPROXY`, `GOPRIVATE`, `GONOSUMCHECK`, `GONOSUMDB`
- [ ] Private module configuration for corporate/private repositories
- [ ] Go toolchain commands: `go build`, `go run`, `go test`, `go vet`, `go fmt`, `go doc`, `go install`, `go get`
- [ ] `go generate` for code generation
- [ ] `go env` to inspect environment variables
- [ ] Cross-compilation: `GOOS`, `GOARCH` (e.g., `GOOS=linux GOARCH=amd64 go build`)
- [ ] Build tags and conditional compilation (`//go:build`)
- [ ] IDE setup: VSCode with Go extension (gopls), GoLand
- [ ] `gopls` (Go language server): features, configuration, troubleshooting
- [ ] Code formatting: `gofmt`, `goimports`
- [ ] Linting: `golangci-lint` (aggregates multiple linters)
- [ ] `staticcheck` for static analysis
- [ ] `go vet` for suspicious code patterns
- [ ] Debugging: `delve` (`dlv`) debugger — breakpoints, stepping, variable inspection
- [ ] Git basics: `init`, `clone`, `add`, `commit`, `push`, `pull`, `branch`, `merge`, `rebase`
- [ ] Git workflows: feature branches, pull requests, conventional commits
- [ ] Makefile basics for Go projects
- [ ] `.editorconfig` for consistent formatting across editors

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | CI/CD pipelines invoke Go toolchain commands. Understanding cross-compilation is essential for building binaries that run on Linux servers from macOS dev machines. |
| **Platform Engineering** | Platform teams maintain internal Go modules and private proxies. Module management and versioning is daily work. |
| **Kubernetes** | Kubernetes, Helm, and kubectl are all written in Go. Contributing to or extending these requires deep toolchain knowledge. |
| **MLOps** | ML infrastructure tools (custom schedulers, pipeline runners) are built in Go. Reproducible builds matter. |
| **LLM Infrastructure** | Building custom inference routers, load balancers, and API gateways requires solid Go project setup. |
| **Cloud Infrastructure** | Terraform, Packer, Consul, Vault — all Go. Understanding their build systems helps when extending them. |
| **GPU ML Systems** | Cross-compiling Go binaries with CGO for GPU-related C libraries requires understanding of build flags and CGO_ENABLED. |
| **Distributed Systems** | Multi-platform binary distribution, reproducible builds, and dependency management are critical for distributed system components. |

### Practical Exercises

- [ ] Install Go from official binaries (not package manager) on Linux/macOS
- [ ] Verify installation: `go version`, `go env`
- [ ] Create your first Go module: `go mod init github.com/yourname/hello`
- [ ] Write a "Hello World" program, build it, and run it
- [ ] Cross-compile the binary for Linux AMD64 and Linux ARM64
- [ ] Set up VSCode with the Go extension, verify `gopls` works
- [ ] Install and run `golangci-lint` on your project
- [ ] Install `delve` and set a breakpoint in your program
- [ ] Create a `Makefile` with targets: `build`, `test`, `lint`, `run`, `clean`
- [ ] Add a third-party dependency (e.g., `github.com/fatih/color`), use it, and run `go mod tidy`
- [ ] Configure `GOPRIVATE` for a hypothetical private module path

### Mini Projects

- [ ] **Go Project Bootstrapper**: Write a shell script that creates a new Go project with standard directory layout (`cmd/`, `internal/`, `pkg/`), initializes a module, creates a Makefile, adds `.gitignore`, and initializes git
- [ ] **Multi-Arch Builder**: Script that cross-compiles a Go binary for linux/amd64, linux/arm64, darwin/amd64, darwin/arm64, and windows/amd64, then checksums each binary
- [ ] **Dependency Auditor**: Script that runs `go list -m all`, checks for known vulnerabilities using `govulncheck`, and generates a report

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Hello World variations (print patterns) | HackerRank | Easy | Get comfortable writing Go code |
| Simple Array Sum | HackerRank | Easy | Basic input/output in Go |
| Compare the Triplets | HackerRank | Easy | Conditionals and loops in Go |
| Diagonal Difference | HackerRank | Easy | 2D arrays, iteration |
| Plus Minus | HackerRank | Easy | Formatting output, floating point |

### Interview Focused Notes

**Common Interview Questions:**
- What is the difference between `GOPATH` and Go modules?
- How does `go mod tidy` work?
- What is `go.sum` and why does it exist?
- How do you handle private modules in a corporate environment?
- Explain cross-compilation in Go. What is CGO_ENABLED and when do you need it?
- What linters do you use and why?

**Common Mistakes:**
- Not running `go mod tidy` before committing
- Committing `go.sum` inconsistencies
- Not understanding that `go get` in module mode differs from GOPATH mode
- Using `go install` when you mean `go build` (and vice versa)
- Not setting `GOPRIVATE` for private modules, causing build failures in CI

**Interviewer Expectations:**
- Can set up a Go project from scratch with proper module configuration
- Understands the Go toolchain deeply, not just surface-level usage
- Can troubleshoot build issues, dependency conflicts, and CI/CD integration
- Knows cross-compilation and platform-specific build considerations

---

---

# STAGE 2: GO LANGUAGE FUNDAMENTALS

---

## 2.1 Go Syntax, Variables, and Constants

### Concepts to Learn

- [ ] Package declaration: `package main`, `package <name>`
- [ ] Import statements: single import, grouped imports, blank imports (`_`), dot imports
- [ ] The `main` function: entry point of a Go program
- [ ] `func init()`: initialization function, execution order, multiple init functions
- [ ] Variable declaration: `var x int`, `var x int = 5`, `x := 5` (short declaration)
- [ ] Multiple variable declaration: `var a, b, c int`
- [ ] Block variable declaration: `var ( ... )`
- [ ] Zero values: every type has a zero value (`0`, `""`, `false`, `nil`, etc.)
- [ ] Constants: `const`, typed constants, untyped constants
- [ ] `iota` for enumerated constants
- [ ] Constant expressions and compile-time evaluation
- [ ] Naming conventions: exported (uppercase) vs unexported (lowercase)
- [ ] Blank identifier `_` for discarding values
- [ ] Shadowing: variable shadowing in inner scopes (a major source of bugs)
- [ ] Type inference with `:=`
- [ ] Explicit type conversion (Go has no implicit casting): `int(x)`, `float64(x)`, `string(x)`

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Configuration constants, iota for enumerated states (deployment stages, environments), variable naming impacts team readability. |
| **Platform Engineering** | Init functions are used in plugin registration patterns. Understanding package initialization order is critical when building extensible platforms. |
| **Kubernetes** | Kubernetes codebase uses `iota` extensively for enumerated types (pod phases, container states). Understanding zero values prevents nil pointer panics. |
| **MLOps** | Constants for model types, pipeline stages, and configuration. Zero values determine default behavior in config structs. |
| **LLM Infrastructure** | Token limits, batch sizes, and inference parameters are often constants. Type safety prevents mixing up int64 token counts with float64 probabilities. |
| **Cloud Infrastructure** | API version constants, resource type enumerations, region/zone constants. |
| **GPU ML Systems** | GPU device IDs, memory limits, CUDA version constants. |
| **Distributed Systems** | Protocol version constants, timeout defaults, retry limits. Variable shadowing bugs in concurrent code are extremely dangerous. |

### Practical Exercises

- [ ] Declare variables using all three methods (`var`, `var =`, `:=`)
- [ ] Create an `iota`-based enumeration for HTTP status categories (Informational, Success, Redirect, ClientError, ServerError)
- [ ] Write a program that demonstrates variable shadowing and explain why the output is surprising
- [ ] Create typed constants that prevent accidentally mixing units (e.g., `type Meters float64`, `type Kilometers float64`)
- [ ] Write a program that shows all zero values for: int, float64, string, bool, slice, map, pointer, interface, channel
- [ ] Use blank imports to understand side effects (e.g., `_ "image/png"`)
- [ ] Demonstrate explicit type conversion between int, float64, and string

### Mini Projects

- [ ] **Config Constants Library**: Create a package that defines all configuration constants for a microservice (ports, timeouts, retry limits, environment names) using `iota` and typed constants
- [ ] **Type-Safe Units**: Build a units package where you cannot accidentally add Meters to Seconds, with conversion functions between related types

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Number of Good Pairs | LeetCode #1512 | Easy | Basic counting with variables |
| Richest Customer Wealth | LeetCode #1672 | Easy | Nested loops, sum variable |
| Shuffle the Array | LeetCode #1470 | Easy | Array index manipulation |
| Kids With the Greatest Number of Candies | LeetCode #1431 | Easy | Comparison, boolean results |
| Subtract the Product and Sum of Digits | LeetCode #1281 | Easy | Arithmetic operations, digit extraction |

### Interview Focused Notes

**Common Interview Questions:**
- What is the zero value in Go and why is it important?
- What is `iota` and how does it work across a const block?
- What is variable shadowing? Give an example where it causes a bug.
- Why does Go not have implicit type conversions?
- What is the difference between `var x int` and `x := 0`?
- What is the blank identifier and when do you use it?
- In what order are init functions and package-level variables initialized?

**Common Mistakes:**
- Variable shadowing with `:=` in if/for blocks (e.g., `err` gets shadowed)
- Assuming `string(65)` gives `"65"` (it actually gives `"A"` — ASCII character)
- Forgetting that short declarations (`:=`) require at least one new variable on the left
- Not understanding that untyped constants have higher precision than typed ones

**Interviewer Expectations:**
- Can explain zero values for all types without hesitation
- Understands the initialization order of a Go program
- Knows the dangers of variable shadowing and how to detect it (linters)
- Can use `iota` for real-world enumerations

---

## 2.2 Data Types (Primitive and Composite)

### Concepts to Learn

- [ ] Boolean: `bool` — `true`, `false`
- [ ] Integers: `int`, `int8`, `int16`, `int32`, `int64`, `uint`, `uint8`, `uint16`, `uint32`, `uint64`, `uintptr`
- [ ] Platform-dependent sizes: `int` and `uint` are 32 or 64 bit depending on platform
- [ ] `byte` (alias for `uint8`), `rune` (alias for `int32`)
- [ ] Floating point: `float32`, `float64` — IEEE 754, precision issues
- [ ] Complex numbers: `complex64`, `complex128` (rarely used but exists)
- [ ] Strings: immutable byte slices, UTF-8 encoded by default
- [ ] String internals: `len()` returns bytes not runes, iterating with `range` yields runes
- [ ] String operations: concatenation, `strings` package (`Contains`, `HasPrefix`, `Split`, `Join`, `Replace`, `TrimSpace`, `ToLower`, `ToUpper`)
- [ ] `strings.Builder` for efficient string concatenation
- [ ] `strconv` package: `Atoi`, `Itoa`, `ParseFloat`, `FormatFloat`, `ParseBool`
- [ ] `fmt` package: `Sprintf`, `Printf`, `Fprintf`, verb specifiers (`%d`, `%s`, `%v`, `%+v`, `%#v`, `%T`, `%p`, `%w`)
- [ ] Type aliases vs type definitions
- [ ] Custom types: `type StatusCode int`, `type Handler func(Request) Response`
- [ ] Type assertions: `val, ok := i.(string)`
- [ ] Type switches: `switch v := i.(type) { ... }`
- [ ] `unsafe.Sizeof` for understanding memory layout (educational, not for production)

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Parsing configuration values (string to int, bool flags). String manipulation for log parsing, template rendering. |
| **Platform Engineering** | Custom types for domain modeling (ResourceName, Namespace, ClusterID). Type safety prevents cross-domain bugs. |
| **Kubernetes** | Kubernetes uses custom types extensively (`resource.Quantity`, `metav1.Time`). Understanding `int32` vs `int64` matters for API compatibility. |
| **MLOps** | Float64 for metrics, loss values. Understanding floating-point precision matters when comparing model metrics. |
| **LLM Infrastructure** | Token IDs (int32), embedding vectors (float32 slices), prompt strings (UTF-8 handling). |
| **Cloud Infrastructure** | Resource IDs (string), ports (uint16), memory sizes (int64 bytes). |
| **GPU ML Systems** | Float16/float32 awareness for model precision, memory calculations with uint64. |
| **Distributed Systems** | Protocol buffer types map to Go types. Understanding size guarantees prevents cross-platform bugs in serialized data. |

### Practical Exercises

- [ ] Write a program that prints the size and range of every integer type
- [ ] Demonstrate the difference between `len("Hello")` and `len("Hello")` (ASCII vs multibyte UTF-8 string like `len("Hello")` is fine)
- [ ] Iterate over a string containing emoji/non-ASCII characters using both index-based and range-based loops — observe the difference
- [ ] Benchmark string concatenation: `+` operator vs `strings.Builder` vs `bytes.Buffer` for 10,000 concatenations
- [ ] Create a custom type `type Celsius float64` and `type Fahrenheit float64` with conversion methods
- [ ] Write a type switch that handles `int`, `string`, `bool`, and `float64` from an `interface{}`
- [ ] Demonstrate floating-point imprecision: `0.1 + 0.2 != 0.3`

### Mini Projects

- [ ] **Type-Safe Configuration Parser**: Parse a YAML/JSON config file into strongly-typed Go structs with custom types (e.g., `type Port uint16`, `type Duration time.Duration`)
- [ ] **String Processing Toolkit**: Build a CLI tool that performs common string operations (base64 encode/decode, URL encode/decode, hash, word count, character frequency)

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Reverse Integer | LeetCode #7 | Medium | Integer overflow awareness |
| Valid Anagram | LeetCode #242 | Easy | Character counting, rune vs byte |
| Roman to Integer | LeetCode #13 | Easy | String parsing, character mapping |
| Longest Common Prefix | LeetCode #14 | Easy | String comparison |
| String to Integer (atoi) | LeetCode #8 | Medium | Type conversion edge cases |

### Interview Focused Notes

**Common Interview Questions:**
- What is the difference between `byte` and `rune`?
- How are strings represented in Go?
- Why is `len("Hello")` sometimes not what you expect?
- What happens when you convert a `string` to `[]byte` and back?
- How do you efficiently concatenate many strings?
- What is the difference between a type alias and a type definition?
- How do type assertions work? What happens if the assertion fails?

**Common Mistakes:**
- Using `len()` on a string expecting character count instead of byte count (use `utf8.RuneCountInString()`)
- Concatenating strings in a loop with `+` instead of `strings.Builder`
- Not handling the `ok` value in type assertions (panics on failure without `ok`)
- Comparing floating-point numbers with `==` instead of epsilon comparison
- Assuming `int` is always 64 bits

**Interviewer Expectations:**
- Deep understanding of string internals and UTF-8
- Knows when to use which integer type and why
- Can explain type assertions and type switches clearly
- Understands memory implications of type choices

---

## 2.3 Structs

### Concepts to Learn

- [ ] Struct definition: `type Person struct { Name string; Age int }`
- [ ] Struct instantiation: literal syntax, named fields, positional fields
- [ ] Zero-value structs: what happens when you declare `var p Person`
- [ ] Pointer to struct: `&Person{Name: "Alice"}`, automatic dereferencing with `.`
- [ ] Anonymous structs: `s := struct{ X int }{X: 5}`
- [ ] Nested structs: structs containing other structs
- [ ] Embedded structs (composition): `type Employee struct { Person; Company string }`
- [ ] Field promotion with embedded structs
- [ ] Struct tags: `json:"name"`, `yaml:"name"`, `db:"name"`, `validate:"required"`
- [ ] Accessing struct tags via reflection (`reflect.TypeOf`, `reflect.StructTag`)
- [ ] Struct comparison: when structs are comparable (all fields must be comparable)
- [ ] Methods on structs: value receivers vs pointer receivers
- [ ] When to use pointer receivers (mutation, large structs, consistency)
- [ ] Constructor pattern: `func NewPerson(name string, age int) *Person`
- [ ] Functional options pattern for flexible constructors
- [ ] Struct alignment and padding (memory layout)
- [ ] `unsafe.Offsetof` and struct field alignment

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Configuration structs, pipeline stage definitions, deployment target structs with JSON/YAML tags. |
| **Platform Engineering** | Domain models for resources, tenants, clusters. Composition is the primary code reuse mechanism. |
| **Kubernetes** | Every Kubernetes resource is a Go struct. CRDs define custom structs. Understanding struct tags is essential for serialization. |
| **MLOps** | Model metadata structs, training job configs, pipeline step definitions. |
| **LLM Infrastructure** | Inference request/response structs, model config structs, prompt template structs. |
| **Cloud Infrastructure** | API request/response models, cloud resource representations, IaC resource structs. |
| **GPU ML Systems** | GPU device info structs, CUDA memory allocation records, training metrics. |
| **Distributed Systems** | Message structs for protocols, state machine representations, cluster membership records. |

### Practical Exercises

- [ ] Define a `Server` struct with fields for Host, Port, TLS, MaxConnections; add JSON tags
- [ ] Create an embedded struct hierarchy: `BaseResource` -> `KubernetesResource` -> `Deployment`
- [ ] Implement both value and pointer receiver methods on a struct; demonstrate the difference
- [ ] Write a constructor using the functional options pattern (`WithPort`, `WithTLS`, etc.)
- [ ] Create an anonymous struct for a one-off JSON response
- [ ] Write a function that reads struct tags using reflection
- [ ] Demonstrate struct alignment: create a struct with poor field ordering, then optimize it

### Mini Projects

- [ ] **Kubernetes Resource Modeler**: Define Go structs that mirror Kubernetes resource types (Pod, Service, Deployment) with proper JSON tags, embedded metadata, and spec/status split
- [ ] **Configuration System**: Build a config system where structs are populated from environment variables, files, and defaults — with struct tags controlling the source

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Design HashMap | LeetCode #706 | Easy | Struct-based data structure |
| Min Stack | LeetCode #155 | Medium | Struct with methods |
| Implement Queue using Stacks | LeetCode #232 | Easy | Struct composition |
| Design Parking System | LeetCode #1603 | Easy | Struct state management |
| Design Browser History | LeetCode #1472 | Medium | Struct with complex state |

### Interview Focused Notes

**Common Interview Questions:**
- What is the difference between a value receiver and a pointer receiver?
- When should you use a pointer receiver?
- What is struct embedding and how does field promotion work?
- What are struct tags and how are they used?
- How does struct alignment affect memory usage?
- What is the functional options pattern?
- Can you compare two structs with `==`?

**Common Mistakes:**
- Using value receivers when mutation is intended
- Mixing value and pointer receivers on the same type (inconsistency)
- Forgetting that embedded struct methods can be "overridden" (shadowed)
- Not aligning struct fields for minimal padding
- Using positional struct literals (breaks when fields are added)

**Interviewer Expectations:**
- Can design clean struct hierarchies using composition
- Knows when to use value vs pointer receivers and can explain why
- Understands serialization tags and their importance in API development
- Can implement the functional options pattern

---

## 2.4 Arrays, Slices, and Maps

### Concepts to Learn

**Arrays:**
- [ ] Array declaration: `var a [5]int`, `a := [5]int{1, 2, 3, 4, 5}`
- [ ] Arrays are fixed-size, value types (copied on assignment)
- [ ] Array comparison with `==`
- [ ] `[...]` syntax for compiler-counted arrays: `a := [...]int{1, 2, 3}`

**Slices:**
- [ ] Slice declaration: `var s []int`, `s := []int{1, 2, 3}`, `s := make([]int, len, cap)`
- [ ] Slice internals: pointer, length, capacity (slice header)
- [ ] Slice from array: `s := a[1:3]` — shares underlying array
- [ ] `append()`: how it works, when it reallocates, growth strategy
- [ ] `copy()`: shallow copy of slice elements
- [ ] Nil slice vs empty slice: `var s []int` vs `s := []int{}`
- [ ] Slice tricks: delete element, insert element, filter in-place, reverse
- [ ] Full slice expression: `a[low:high:max]` to control capacity
- [ ] Memory leaks with slices (referencing large underlying array)
- [ ] `slices` package (Go 1.21+): `slices.Sort`, `slices.Contains`, `slices.Index`
- [ ] Multi-dimensional slices: `[][]int`

**Maps:**
- [ ] Map declaration: `var m map[string]int`, `m := map[string]int{}`, `m := make(map[string]int)`
- [ ] Map operations: insert, update, delete (`delete(m, key)`), lookup
- [ ] Comma-ok idiom: `val, ok := m[key]`
- [ ] Map iteration order is random (by design)
- [ ] Map internals: hash table, buckets, growth
- [ ] Nil map vs empty map behavior (nil map panics on write)
- [ ] Maps are not safe for concurrent use (need `sync.Map` or mutex)
- [ ] Map of structs: `map[string]Person` — cannot modify fields in-place (use pointers or reassign)
- [ ] Using maps as sets: `map[string]struct{}` (zero memory for values)
- [ ] `maps` package (Go 1.21+): `maps.Keys`, `maps.Values`, `maps.Clone`

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Slices for ordered lists (servers, tasks, steps). Maps for key-value lookups (environment variables, labels, tags). |
| **Platform Engineering** | Resource registries (maps), ordered pipelines (slices), label selectors (map operations). |
| **Kubernetes** | Labels and annotations are `map[string]string`. Container lists, volume mounts, env vars are slices. Understanding slice capacity matters for large resource lists. |
| **MLOps** | Feature vectors (slices), hyperparameter maps, batch processing with slices. |
| **LLM Infrastructure** | Token ID slices, embedding vectors, vocabulary maps, prompt template variables (maps). |
| **Cloud Infrastructure** | Resource tag maps, IP address lists, security group rule slices. |
| **GPU ML Systems** | GPU device lists, memory allocation maps, tensor shape slices. |
| **Distributed Systems** | Routing tables (maps), node lists (slices), consistent hashing (sorted slices). |

### Practical Exercises

- [ ] Demonstrate that arrays are value types (modify a copy, original unchanged)
- [ ] Create a slice, append elements, and print length/capacity at each step to observe growth
- [ ] Demonstrate the shared backing array bug: slice from array, modify slice, observe array change
- [ ] Implement a function that removes an element from a slice by index (preserving and not preserving order)
- [ ] Build a word frequency counter using `map[string]int`
- [ ] Demonstrate nil slice vs empty slice behavior with `json.Marshal`
- [ ] Use a map as a set to find unique elements in a slice
- [ ] Demonstrate the map concurrent access panic with goroutines (then fix with `sync.Mutex`)
- [ ] Use `copy()` to create a true independent copy of a slice
- [ ] Create a 2D slice (matrix) and perform row/column operations

### Mini Projects

- [ ] **In-Memory Key-Value Store**: Build a simple key-value store using maps with support for Get, Set, Delete, List, and TTL expiration
- [ ] **Label Selector Engine**: Implement Kubernetes-style label matching (`matchLabels`, `matchExpressions`) using maps and slices
- [ ] **Sorted Unique Collection**: Implement a sorted set using slices with binary search for O(log n) lookups

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Two Sum | LeetCode #1 | Easy | Map for O(1) lookups |
| Contains Duplicate | LeetCode #217 | Easy | Map as set |
| Intersection of Two Arrays II | LeetCode #350 | Easy | Map counting |
| Group Anagrams | LeetCode #49 | Medium | Map with sorted string key |
| Top K Frequent Elements | LeetCode #347 | Medium | Map + sorting/heap |
| Subarray Sum Equals K | LeetCode #560 | Medium | Prefix sum with map |
| Product of Array Except Self | LeetCode #238 | Medium | Slice manipulation |
| 3Sum | LeetCode #15 | Medium | Sorting + two pointers on slices |

### Interview Focused Notes

**Common Interview Questions:**
- What is the difference between an array and a slice in Go?
- Explain slice internals: what does the slice header contain?
- When does `append` allocate a new underlying array?
- What is the difference between a nil slice and an empty slice?
- How do you safely use maps with concurrent goroutines?
- Why is map iteration order random in Go?
- How can slices cause memory leaks?
- What is `map[string]struct{}` and when would you use it?

**Common Mistakes:**
- Appending to a slice in a function without returning the new slice (caller doesn't see the change if reallocation happened)
- Forgetting to check `ok` in map lookups (using zero value by accident)
- Writing to a nil map (panic)
- Modifying a slice during range iteration (index bugs)
- Not understanding that map values are not addressable (can't do `m["key"].Field = val` with struct values)

**Interviewer Expectations:**
- Can draw the slice header (ptr, len, cap) and explain append behavior
- Knows the gotchas of shared backing arrays
- Understands map internals at a high level
- Can solve medium LeetCode problems using maps and slices efficiently
- Knows concurrent map access restrictions

---

## 2.5 Pointers

### Concepts to Learn

- [ ] What is a pointer: memory address of a value
- [ ] Pointer declaration: `var p *int`
- [ ] Address-of operator: `&x`
- [ ] Dereference operator: `*p`
- [ ] Zero value of a pointer: `nil`
- [ ] Nil pointer dereference panic
- [ ] Pointer to struct: automatic dereferencing (`p.Name` instead of `(*p).Name`)
- [ ] Pointers as function parameters: passing by reference
- [ ] When to use pointers: mutation, large structs, optional values (nil = absent)
- [ ] When NOT to use pointers: small value types, immutability is desired
- [ ] Pointer receivers vs value receivers (revisited)
- [ ] Pointers and escape analysis: stack vs heap allocation
- [ ] No pointer arithmetic in Go (safety feature; `unsafe.Pointer` exists but discouraged)
- [ ] `new()` function: allocates and returns a pointer
- [ ] `new` vs `&T{}` — functionally equivalent
- [ ] Pointers in slices and maps: `[]*Person`, `map[string]*Config`
- [ ] Double pointers: `**int` (rare but exists)
- [ ] `unsafe.Pointer` for low-level interop (CGO, memory-mapped I/O)

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Config objects passed by pointer for mutation. Understanding nil pointers prevents runtime panics in automation tools. |
| **Platform Engineering** | Pointer receivers on service structs. Optional configuration fields using `*string`, `*int`. |
| **Kubernetes** | API objects are always passed as pointers. CRD specs use `*int32` for optional fields. Understanding pointer semantics is essential for controllers. |
| **MLOps** | Large tensor metadata passed by pointer. Pipeline step configurations use pointer fields for optionality. |
| **LLM Infrastructure** | Request/response objects passed by pointer for performance. Model config with optional overrides. |
| **Cloud Infrastructure** | AWS SDK uses `*string` extensively for optional fields. Understanding this pattern is critical. |
| **GPU ML Systems** | GPU memory addresses, device pointers in CGO interop. |
| **Distributed Systems** | Shared state management with pointers, careful nil checking in distributed protocols. |

### Practical Exercises

- [ ] Write a swap function using pointers
- [ ] Demonstrate the difference between passing a struct by value vs by pointer
- [ ] Create a function that returns a pointer to a local variable (demonstrate escape analysis with `go build -gcflags='-m'`)
- [ ] Build a linked list using pointers (Node struct with `Next *Node`)
- [ ] Demonstrate nil pointer dereference and guard against it
- [ ] Use `*string` for an optional field in a JSON struct (distinguish between absent and empty)
- [ ] Write a function that modifies a slice through a pointer vs returning a new slice

### Mini Projects

- [ ] **Linked List Library**: Implement singly and doubly linked lists with Insert, Delete, Search, Reverse operations
- [ ] **Optional Config Builder**: Build a config system where optional fields use pointers, with helper functions like `StringPtr(s string) *string`

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Reverse Linked List | LeetCode #206 | Easy | Pointer manipulation |
| Linked List Cycle | LeetCode #141 | Easy | Two-pointer with pointers |
| Merge Two Sorted Lists | LeetCode #21 | Easy | Pointer reassignment |
| Remove Nth Node From End | LeetCode #19 | Medium | Two-pointer technique |
| Add Two Numbers | LeetCode #2 | Medium | Linked list traversal |
| Copy List with Random Pointer | LeetCode #138 | Medium | Deep copy with pointer mapping |

### Interview Focused Notes

**Common Interview Questions:**
- Does Go have pass-by-reference?
- What is the zero value of a pointer?
- When should you use a pointer receiver?
- What is escape analysis and how does it relate to pointers?
- Why does Go not have pointer arithmetic?
- What is `unsafe.Pointer` and when would you use it?
- How do optional fields work in Go (pointer pattern)?

**Common Mistakes:**
- Returning a pointer to a loop variable (all pointers point to the same variable — fixed in Go 1.22)
- Not checking for nil before dereferencing
- Using pointers everywhere "for performance" when value types are fine (premature optimization)
- Confusion about pointer receiver method sets (interface satisfaction)

**Interviewer Expectations:**
- Can explain stack vs heap allocation and escape analysis
- Understands when pointers improve performance vs when they don't
- Can work with pointer-based data structures (linked lists, trees)
- Knows the optional field pattern with `*T`

---

## 2.6 Control Flow

### Concepts to Learn

- [ ] `if` statement: no parentheses required, braces mandatory
- [ ] `if` with initialization: `if val, err := doSomething(); err != nil { ... }`
- [ ] `else if` and `else` chains
- [ ] `for` loop: Go's only loop construct
- [ ] Classic for loop: `for i := 0; i < n; i++ { ... }`
- [ ] While-style for loop: `for condition { ... }`
- [ ] Infinite loop: `for { ... }`
- [ ] `for range` over slices, maps, strings, channels
- [ ] Range loop variables: index, value (copies for slices; key, value for maps)
- [ ] Range loop variable capture bug (pre-Go 1.22) and fix
- [ ] `break`, `continue`, `goto`
- [ ] Labeled loops and labeled `break`/`continue`
- [ ] `switch` statement: no fallthrough by default (unlike C)
- [ ] `switch` with no condition (cleaner than if-else chains)
- [ ] `switch` with initialization
- [ ] `fallthrough` keyword (explicit, rarely used)
- [ ] Type switch: `switch v := x.(type) { ... }`
- [ ] `defer` statement: LIFO execution, common patterns (cleanup, unlock, close)
- [ ] `defer` with closures: capturing variables
- [ ] `defer` in loops (resource leak gotcha)
- [ ] `panic` and `recover`: when to use (almost never in application code)
- [ ] `panic` for programming errors, not user errors

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Switch statements for handling different deployment environments/stages. Defer for cleanup in automation scripts. |
| **Platform Engineering** | Range over resource lists, switch on resource types, defer for releasing locks. |
| **Kubernetes** | Reconciliation loops are `for` loops. Switch on resource events. Defer is everywhere for cleanup. |
| **MLOps** | Pipeline step iteration, switch on step types, defer for resource cleanup (GPU memory, file handles). |
| **LLM Infrastructure** | Batch processing loops, switch on model types/backends, defer for connection cleanup. |
| **Cloud Infrastructure** | Pagination loops (for range with API responses), switch on cloud provider. |
| **GPU ML Systems** | Iteration over GPU devices, switch on device capabilities, defer for GPU memory release. |
| **Distributed Systems** | Event processing loops, type switches on message types, defer for connection cleanup, labeled breaks in nested select/for. |

### Practical Exercises

- [ ] Write an `if` with initialization that opens a file and immediately checks the error
- [ ] Implement FizzBuzz using a switch with no condition
- [ ] Demonstrate labeled break in a nested loop (exit outer loop from inner)
- [ ] Write a function with multiple defers and verify LIFO order
- [ ] Demonstrate the defer-in-loop resource leak and fix it with an inner function
- [ ] Write a safe division function using `panic`/`recover`
- [ ] Use a type switch to handle multiple types from an `interface{}`
- [ ] Implement a retry loop with exponential backoff using `for` and `time.Sleep`

### Mini Projects

- [ ] **Command Dispatcher**: Build a CLI command dispatcher using switch statements that routes commands to handler functions
- [ ] **Retry Framework**: Implement a generic retry function with configurable max attempts, backoff strategy, and error classification

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Palindrome Number | LeetCode #9 | Easy | Loops and conditionals |
| Power of Two | LeetCode #231 | Easy | Bit manipulation with conditionals |
| Missing Number | LeetCode #268 | Easy | Loop with math/XOR |
| Single Number | LeetCode #136 | Easy | XOR in a loop |
| Climbing Stairs | LeetCode #70 | Easy | Loop-based DP |
| Binary Search | LeetCode #704 | Easy | While loop with pointers |

### Interview Focused Notes

**Common Interview Questions:**
- Does Go have a while loop? (No, only `for`)
- How does `defer` work? In what order do deferred functions execute?
- What happens if you defer in a loop?
- When should you use `panic`? When should you NOT?
- What is the difference between `break` and `continue`?
- How do labeled loops work?
- What is the range loop variable capture bug?

**Common Mistakes:**
- Deferring in a loop (defers don't execute until function returns — resource leak)
- Using `panic` for error handling instead of returning errors
- Forgetting that `switch` in Go does NOT fall through by default
- Range loop variable capture (pre-Go 1.22): closure captures the loop variable, not the value

**Interviewer Expectations:**
- Knows that Go has only one loop construct
- Can explain `defer` execution order and common patterns
- Understands `panic`/`recover` and when they're appropriate
- Can write clean control flow without unnecessary nesting

---

## 2.7 Functions

### Concepts to Learn

- [ ] Function declaration: `func name(params) returnType { ... }`
- [ ] Multiple parameters: `func add(a, b int) int`
- [ ] Multiple return values: `func divide(a, b float64) (float64, error)`
- [ ] Named return values: `func divide(a, b float64) (result float64, err error)`
- [ ] Naked return (using named return values) — generally discouraged in long functions
- [ ] Variadic functions: `func sum(nums ...int) int`
- [ ] Spreading a slice into variadic: `sum(numbers...)`
- [ ] First-class functions: functions as values, variables, and arguments
- [ ] Anonymous functions (function literals / lambdas)
- [ ] Closures: functions that capture outer variables
- [ ] Closure variable capture: by reference, not by value
- [ ] Higher-order functions: functions that take or return functions
- [ ] Common functional patterns: `map`, `filter`, `reduce` (manual implementation)
- [ ] Method syntax: `func (s *Server) Start() error`
- [ ] Function types: `type HandlerFunc func(http.ResponseWriter, *http.Request)`
- [ ] Recursive functions and tail recursion (Go does NOT optimize tail calls)
- [ ] `init()` functions: special initialization, called before `main()`
- [ ] Blank functions for interface satisfaction

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Callback functions for event handlers, middleware chains, variadic options for CLI tools. |
| **Platform Engineering** | Handler functions, middleware stacks, functional options pattern for service configuration. |
| **Kubernetes** | Reconciler functions, admission webhook handlers, informer callbacks. Multiple return values (result, error) is the universal pattern. |
| **MLOps** | Pipeline step functions, data transformation functions, callback hooks for training events. |
| **LLM Infrastructure** | Request handler functions, middleware for auth/rate-limiting, streaming response closures. |
| **Cloud Infrastructure** | Provider factory functions, resource handler callbacks, retry function wrappers. |
| **GPU ML Systems** | Device allocation functions, cleanup closures for GPU resources, callback functions for training progress. |
| **Distributed Systems** | RPC handler functions, message processing callbacks, consensus algorithm step functions. |

### Practical Exercises

- [ ] Write a function that returns multiple values (result + error)
- [ ] Implement a variadic `max` function that finds the maximum of any number of ints
- [ ] Create a higher-order function `retry(attempts int, fn func() error) error`
- [ ] Implement `map`, `filter`, `reduce` for `[]int` using higher-order functions
- [ ] Demonstrate closure variable capture: create a counter function using closures
- [ ] Write a function that returns a function (function factory pattern)
- [ ] Create a middleware chain using function composition
- [ ] Demonstrate the difference between named and unnamed return values with defer

### Mini Projects

- [ ] **Middleware Engine**: Build an HTTP middleware system where middlewares are functions that wrap handlers: `type Middleware func(http.Handler) http.Handler`
- [ ] **Pipeline Builder**: Create a data pipeline where each stage is a function: `type Stage func(input interface{}) (output interface{}, error)`
- [ ] **Event System**: Build a pub-sub event system using function callbacks: `On(event string, handler func(data interface{}))`, `Emit(event string, data interface{})`

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Fibonacci Number | LeetCode #509 | Easy | Recursion basics |
| Power(x, n) | LeetCode #50 | Medium | Recursive divide and conquer |
| Generate Parentheses | LeetCode #22 | Medium | Backtracking with recursion |
| Letter Combinations of a Phone Number | LeetCode #17 | Medium | Recursion with string building |
| Permutations | LeetCode #46 | Medium | Backtracking fundamentals |
| Flatten Nested List Iterator | LeetCode #341 | Medium | Recursive data structure handling |

### Interview Focused Notes

**Common Interview Questions:**
- How do multiple return values work in Go?
- What is a closure? Give a practical example.
- What is the functional options pattern?
- What are variadic functions and how do you use them?
- What is the difference between a function and a method in Go?
- Can Go functions be passed as arguments? Give an example.
- What are init functions and in what order do they execute?

**Common Mistakes:**
- Closure capturing loop variable by reference (getting the last value)
- Using naked returns in long functions (reduces readability)
- Not using the error return value pattern consistently
- Recursive functions without proper base cases (stack overflow)
- Creating closures in goroutines without proper variable capture

**Interviewer Expectations:**
- Can explain closures and demonstrate with practical examples
- Knows the error return pattern and uses it consistently
- Can implement functional patterns (middleware, pipeline, options)
- Understands function types and higher-order functions

---

## 2.8 Error Handling

### Concepts to Learn

- [ ] The `error` interface: `type error interface { Error() string }`
- [ ] Creating errors: `errors.New("something failed")`, `fmt.Errorf("failed: %w", err)`
- [ ] Error checking pattern: `if err != nil { return err }`
- [ ] Error wrapping: `fmt.Errorf("operation failed: %w", err)` (Go 1.13+)
- [ ] Error unwrapping: `errors.Unwrap(err)`
- [ ] Error comparison: `errors.Is(err, target)` — checks error chain
- [ ] Error type assertion: `errors.As(err, &target)` — checks error chain for type
- [ ] Sentinel errors: `var ErrNotFound = errors.New("not found")`
- [ ] Custom error types: `type ValidationError struct { Field, Message string }`
- [ ] Implementing the `error` interface on custom types
- [ ] Error handling strategies: return, wrap, log-and-return, handle-and-continue
- [ ] Don't ignore errors: `_ = file.Close()` — acceptable ONLY when you truly don't care
- [ ] Error handling anti-patterns: `panic` for errors, logging and returning (double reporting)
- [ ] The `%w` verb vs `%v` for error wrapping (only `%w` allows `errors.Is`/`errors.As`)
- [ ] Multi-error handling: `errors.Join()` (Go 1.20+)
- [ ] Error handling in concurrent code: error channels, `errgroup`
- [ ] `golang.org/x/sync/errgroup` for concurrent error handling
- [ ] Idiomatic error messages: lowercase, no punctuation, no prefix

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Every CLI tool, every API call, every file operation can fail. Proper error handling is the difference between a tool that's debuggable and one that's a nightmare. |
| **Platform Engineering** | Error propagation through service layers. Custom error types for domain-specific failures. Error wrapping for context. |
| **Kubernetes** | Controller reconciliation must handle every possible error. Status conditions report errors. Admission webhooks return structured errors. |
| **MLOps** | Training failures, data pipeline errors, model validation failures — all need proper error classification and handling. |
| **LLM Infrastructure** | Token limit errors, model loading failures, inference timeouts, rate limit errors — each needs distinct handling. |
| **Cloud Infrastructure** | API errors from cloud providers, retry logic for transient errors, error classification for monitoring. |
| **GPU ML Systems** | CUDA errors, out-of-memory errors, device initialization failures — each needs specific error types. |
| **Distributed Systems** | Network partition errors, consensus failures, timeout errors — proper error propagation is critical for system reliability. |

### Practical Exercises

- [ ] Create a function that returns wrapped errors through three call layers; use `errors.Is` to check
- [ ] Define a custom `APIError` type with `StatusCode`, `Message`, `RequestID`; use `errors.As` to extract
- [ ] Define sentinel errors for a user service: `ErrUserNotFound`, `ErrUserAlreadyExists`, `ErrInvalidEmail`
- [ ] Write an `errgroup` example that fetches 5 URLs concurrently and collects errors
- [ ] Demonstrate the difference between wrapping with `%w` and `%v`
- [ ] Use `errors.Join` to combine multiple validation errors into one
- [ ] Write a function that classifies errors as retryable vs non-retryable

### Mini Projects

- [ ] **Error Handling Library**: Build a custom error package with `New`, `Wrap`, `Is`, `As`, `WithCode`, `WithMetadata`, and `Stack` (capture stack trace)
- [ ] **Resilient HTTP Client**: Build an HTTP client wrapper that classifies errors (network, timeout, 4xx, 5xx), retries transient errors, and wraps all errors with context

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Valid Sudoku | LeetCode #36 | Medium | Validation with error-like returns |
| Decode String | LeetCode #394 | Medium | Parsing with error conditions |
| Evaluate Reverse Polish Notation | LeetCode #150 | Medium | Error conditions in computation |
| Basic Calculator II | LeetCode #227 | Medium | Parsing and error handling |

### Interview Focused Notes

**Common Interview Questions:**
- How does error handling work in Go compared to exceptions in other languages?
- What is error wrapping and why is it important?
- What is the difference between `errors.Is` and `errors.As`?
- What are sentinel errors? When would you use them vs custom error types?
- How do you handle errors in concurrent Go code?
- What is `errgroup` and how does it work?
- What is the idiomatic way to format error messages in Go?

**Common Mistakes:**
- Using `panic` instead of returning errors
- Logging an error AND returning it (double reporting)
- Using `%v` instead of `%w` when wrapping (breaks `errors.Is`/`errors.As`)
- Not wrapping errors with context (bare `return err` through many layers = undebuggable)
- Comparing errors with `==` instead of `errors.Is` (breaks with wrapped errors)
- Error messages starting with capital letters or ending with punctuation

**Interviewer Expectations:**
- Follows the `if err != nil` pattern consistently
- Always wraps errors with context
- Knows when to use sentinel errors vs custom types
- Can handle errors in concurrent code
- Error messages are lowercase and descriptive

---

## 2.9 Packages and Modules

### Concepts to Learn

- [ ] Package concept: grouping related code, one package per directory
- [ ] Package naming: lowercase, short, no underscores, matches directory name
- [ ] Export rules: uppercase = exported, lowercase = unexported
- [ ] `internal` packages: restricted visibility, Go enforces import restrictions
- [ ] Package initialization order: imported packages first, then `var`, then `init()`
- [ ] Circular import prevention (Go forbids circular dependencies)
- [ ] Standard library overview: key packages you must know
  - [ ] `fmt`, `os`, `io`, `bufio`, `strings`, `strconv`
  - [ ] `net/http`, `encoding/json`, `encoding/xml`
  - [ ] `sync`, `context`, `time`, `math`, `sort`
  - [ ] `path/filepath`, `os/exec`, `log`, `flag`
  - [ ] `testing`, `net/http/httptest`
  - [ ] `crypto/*`, `hash/*`
  - [ ] `reflect`, `unsafe` (know they exist, use sparingly)
- [ ] Module system: `go.mod`, `go.sum`
- [ ] Module path: typically matches repository URL
- [ ] Semantic import versioning: `v0`, `v1` (implicit), `v2+` (path suffix)
- [ ] `go mod init`, `go mod tidy`, `go mod vendor`, `go mod download`, `go mod graph`
- [ ] `go.sum` and checksum verification
- [ ] Vendoring: `go mod vendor` and `go build -mod=vendor`
- [ ] Replace directives: `replace` in `go.mod` for local development
- [ ] Workspace mode: `go.work` for multi-module development (Go 1.18+)
- [ ] Module proxies: `proxy.golang.org`, `GOPROXY` setting
- [ ] Private modules: `GOPRIVATE`, `GONOSUMCHECK`
- [ ] Go standard project layout (community convention, not official)
  - [ ] `cmd/` for main packages
  - [ ] `internal/` for private packages
  - [ ] `pkg/` for public library code (debated)
  - [ ] `api/` for API definitions (protobuf, OpenAPI)
  - [ ] `configs/`, `scripts/`, `build/`, `deployments/`, `docs/`, `test/`

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | CLI tools use standard project layout. Internal packages protect domain logic. Module management for private tooling. |
| **Platform Engineering** | Multi-module monorepos, workspace mode, internal SDK packages, versioned APIs. |
| **Kubernetes** | Kubernetes uses the standard project layout. client-go is a module with strict versioning. Understanding module compatibility is critical. |
| **MLOps** | Internal ML libraries, versioned model serving SDKs, shared pipeline packages. |
| **LLM Infrastructure** | Shared inference client libraries, internal prompt engineering packages, versioned API clients. |
| **Cloud Infrastructure** | Terraform providers follow specific project layouts. Private modules for internal infrastructure. |
| **GPU ML Systems** | Internal GPU management libraries, shared device plugins. |
| **Distributed Systems** | Shared protocol packages, versioned client libraries, internal consensus implementations. |

### Practical Exercises

- [ ] Create a multi-package project with `cmd/`, `internal/`, and `pkg/` directories
- [ ] Create an `internal` package and verify it can't be imported from outside
- [ ] Set up two modules locally and use `replace` directive for local development
- [ ] Use `go.work` to manage a workspace with multiple modules
- [ ] Add a v2 of a library with the `/v2` path suffix
- [ ] Use `go mod graph` to visualize your dependency tree
- [ ] Configure `GOPRIVATE` for a hypothetical private module and verify with `go env`

### Mini Projects

- [ ] **Go Project Template**: Create a reusable project template with the standard layout, Makefile, Dockerfile, CI config, linter config, and README
- [ ] **Internal SDK**: Build a multi-package internal SDK with domain models, client library, and utilities, demonstrating proper package boundaries

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Implement Trie | LeetCode #208 | Medium | Package-worthy data structure |
| Design Add and Search Words Data Structure | LeetCode #211 | Medium | Building reusable components |
| LRU Cache | LeetCode #146 | Medium | Reusable data structure design |
| Serialize and Deserialize Binary Tree | LeetCode #297 | Hard | Encoding/decoding — relates to encoding packages |

### Interview Focused Notes

**Common Interview Questions:**
- What is the difference between a package and a module?
- How does Go prevent circular imports?
- What is the `internal` package convention?
- How does semantic import versioning work in Go?
- What is `go.sum` and what problem does it solve?
- How do you manage private dependencies in a corporate environment?
- Describe a Go project layout you would use for a microservice.
- What is `go.work` and when would you use it?

**Common Mistakes:**
- Package names that are too generic (`util`, `common`, `helpers`) — be specific
- Circular dependencies (restructure packages or use interfaces)
- Not running `go mod tidy` before committing
- Importing `v2` of a module without the `/v2` suffix
- Putting everything in one package (no separation of concerns)

**Interviewer Expectations:**
- Can design a clean project layout for a production microservice
- Understands module versioning and compatibility guarantees
- Knows how to handle private modules and vendoring
- Can explain package initialization order

---

---

# STAGE 3: GO IDIOMS AND BEST PRACTICES

---

## 3.1 Idiomatic Go

### Concepts to Learn

- [ ] Go proverbs (Rob Pike's Go Proverbs — know them all):
  - "Don't communicate by sharing memory; share memory by communicating"
  - "Concurrency is not parallelism"
  - "Channels orchestrate; mutexes serialize"
  - "The bigger the interface, the weaker the abstraction"
  - "Make the zero value useful"
  - "interface{} says nothing"
  - "Gofmt's style is no one's favorite, yet gofmt is everyone's favorite"
  - "A little copying is better than a little dependency"
  - "Clear is better than clever"
  - "Errors are values"
  - "Don't just check errors, handle them gracefully"
  - "Design the architecture, name the components, document the details"
  - "Documentation is for users"
- [ ] Effective Go (official guide) — key principles:
  - Formatting with `gofmt`
  - Commentary: godoc conventions, package comments, function comments
  - Names: MixedCaps, acronyms (HTTP, URL, ID), short names in small scopes
  - Semicolons: automatic insertion rules (why opening braces must be on the same line)
  - Control structures: prefer early returns (guard clauses), reduce nesting
  - Functions: error returns, named results, defer
  - Data: allocation with `new` vs `make`, arrays/slices/maps
  - Methods: pointer vs value receivers, interfaces
  - Interfaces: accept interfaces, return structs
  - Embedding: composition over inheritance
  - Concurrency: goroutines, channels, select
  - Errors: the error type, custom errors
- [ ] Code style guidelines from `golang.org/wiki/CodeReviewComments`:
  - Don't use `else` when the `if` block ends with `return`
  - Handle errors first (guard clause pattern)
  - Don't use `panic` in library code
  - Use `context.Context` as the first parameter
  - Avoid package-level state (global variables)
  - Variable names: short in small scopes, descriptive in larger scopes
  - Receiver names: short, consistent, not `this` or `self`
  - Error strings: lowercase, no period
  - Package comments: before the package declaration in any one file
  - Test file naming: `*_test.go`

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Code reviews in Go teams expect idiomatic code. Non-idiomatic Go is rejected in PRs. |
| **Platform Engineering** | Platform SDKs must follow Go idioms or users will reject them. API design follows "accept interfaces, return structs." |
| **Kubernetes** | The entire Kubernetes codebase follows these idioms. Contributing requires matching the style. |
| **MLOps** | ML infrastructure Go code reviewed by Go experts must be idiomatic. |
| **LLM Infrastructure** | Same — any Go code in production will be reviewed against these standards. |
| **Cloud Infrastructure** | HashiCorp, Docker, and other infra tools follow these idioms strictly. |
| **GPU ML Systems** | Same standards apply for any Go code interfacing with GPU management. |
| **Distributed Systems** | Distributed systems code is reviewed more carefully — idiomatic code is more maintainable. |

### Practical Exercises

- [ ] Take a non-idiomatic Go program (deeply nested, poor naming, no error handling) and refactor it to be idiomatic
- [ ] Write a package with proper godoc comments and generate documentation
- [ ] Refactor a function with 5+ levels of nesting to use guard clauses (early returns)
- [ ] Review a Go code sample and list all non-idiomatic patterns
- [ ] Write an interface-first design: define the interface, then the struct that implements it
- [ ] Implement "accept interfaces, return structs" in a real service layer

### Mini Projects

- [ ] **Code Review Bot**: Write a Go program that parses Go source files and flags common non-idiomatic patterns (using `go/ast` and `go/parser`)
- [ ] **Idiomatic Refactoring Exercise Set**: Create a repository with 10 intentionally non-idiomatic Go programs and their idiomatic solutions

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Valid Palindrome | LeetCode #125 | Easy | Clean, idiomatic two-pointer |
| Implement strStr() | LeetCode #28 | Easy | Clean string processing |
| Remove Duplicates from Sorted Array | LeetCode #26 | Easy | Idiomatic in-place modification |
| Best Time to Buy and Sell Stock | LeetCode #121 | Easy | Clean single-pass algorithm |
| Maximum Depth of Binary Tree | LeetCode #104 | Easy | Clean recursive vs iterative |

### Interview Focused Notes

**Common Interview Questions:**
- What does "accept interfaces, return structs" mean?
- What does "make the zero value useful" mean? Give examples.
- Why does Go prefer composition over inheritance?
- What are Go proverbs and which ones do you follow?
- How do you structure a Go project?
- What is the guard clause pattern?

**Common Mistakes:**
- Writing Java/Python-style Go (classes, getters/setters, deep inheritance)
- Using `this` or `self` as receiver names
- Returning interfaces instead of concrete types
- Over-abstracting with too many interfaces
- Not following `gofmt` (instant PR rejection)

**Interviewer Expectations:**
- Code looks like it was written by a Go developer, not a Java developer
- Follows all formatting and naming conventions
- Error handling is idiomatic (no panics, proper wrapping)
- Clean, readable code with minimal nesting

---

## 3.2 Zero Values and Their Power

### Concepts to Learn

- [ ] Zero value for every type:
  - `int` → `0`
  - `float64` → `0.0`
  - `bool` → `false`
  - `string` → `""`
  - `pointer` → `nil`
  - `slice` → `nil`
  - `map` → `nil`
  - `channel` → `nil`
  - `interface` → `nil`
  - `struct` → all fields are zero-valued
  - `function` → `nil`
- [ ] "Make the zero value useful" principle:
  - `sync.Mutex` — zero value is an unlocked mutex (no constructor needed)
  - `bytes.Buffer` — zero value is an empty buffer ready to write
  - `sync.WaitGroup` — zero value is ready to use
  - `sync.Once` — zero value is ready to use
- [ ] Designing your own types with useful zero values
- [ ] `nil` slice behavior: `len(nil)` = 0, `append(nil, ...)` works, `json.Marshal(nil)` = `null`
- [ ] `nil` map behavior: reads return zero value, writes panic
- [ ] `nil` interface vs interface holding nil pointer (the infamous `nil` interface gotcha)
- [ ] `nil` channel behavior: send blocks forever, receive blocks forever, close panics
- [ ] `nil` function: calling panics, but can be checked with `if fn != nil`
- [ ] Using zero values to simplify constructors (no NewFoo needed if zero value works)

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Zero-value-ready types simplify configuration. CLI flags default to zero values. |
| **Platform Engineering** | Designing APIs where zero values are valid defaults reduces configuration burden. |
| **Kubernetes** | Kubernetes API objects heavily rely on zero values as defaults. Defaulting webhooks fill in zero values. |
| **MLOps** | Model configs with sensible zero-value defaults. Pipeline steps that work without explicit configuration. |
| **LLM Infrastructure** | Inference parameters with zero-value defaults (temperature=0 means greedy). |
| **Cloud Infrastructure** | Resource defaults, optional configuration, zero-value-based feature detection. |
| **GPU ML Systems** | Device configs where zero values mean "auto-detect." |
| **Distributed Systems** | Protocol messages where zero values are valid defaults. State machines starting from zero state. |

### Practical Exercises

- [ ] Design a `Logger` struct where the zero value logs to stdout with INFO level
- [ ] Design a `RateLimiter` struct where the zero value means "no rate limiting"
- [ ] Demonstrate the nil interface gotcha: `var err *MyError; var e error = err; e != nil` is TRUE
- [ ] Design a `Config` struct where all zero values are sensible defaults
- [ ] Show that `sync.Mutex{}` works without initialization
- [ ] Demonstrate nil slice vs nil map behavior differences

### Mini Projects

- [ ] **Zero-Value-Ready HTTP Server**: Build a simple HTTP server struct where `var s Server; s.ListenAndServe()` works with sensible defaults (port 8080, default handler, default timeouts)
- [ ] **Self-Configuring Worker Pool**: Build a worker pool where zero value means "use runtime.NumCPU() workers" and "unbuffered work channel"

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Design HashSet | LeetCode #705 | Easy | Zero value for set (empty set) |
| Implement Stack using Queues | LeetCode #225 | Easy | Zero value of stack (empty stack) |
| Design Circular Queue | LeetCode #622 | Medium | Initial state as zero value |
| Range Sum Query - Immutable | LeetCode #303 | Easy | Prefix sum with zero default |

### Interview Focused Notes

**Common Interview Questions:**
- What is the zero value in Go? List zero values for all basic types.
- What does "make the zero value useful" mean? Give examples from the standard library.
- What is the nil interface gotcha? Explain with code.
- What is the difference between a nil slice and an empty slice?
- Can you read from a nil map? Can you write to one?
- What happens if you send to a nil channel?

**Common Mistakes:**
- Not designing types with useful zero values (requiring explicit constructors for everything)
- The nil interface gotcha (function returns `*MyError(nil)` assigned to `error` interface — `err != nil` is true)
- Writing to a nil map (panic)
- Closing a nil channel (panic)
- Assuming `nil` slice and `[]T{}` serialize the same way (they don't in JSON: `null` vs `[]`)

**Interviewer Expectations:**
- Can immediately list zero values for all types
- Understands and can explain the nil interface gotcha
- Designs types where zero values work correctly
- Knows the behavioral differences between nil and empty for slices, maps, channels

---

## 3.3 Error Handling Patterns (Advanced)

### Concepts to Learn

- [ ] Error handling philosophy: errors are values, handle them explicitly
- [ ] Error wrapping chain: creating a chain of context through call layers
- [ ] Pattern: wrap at each level with relevant context
  ```
  // Good
  return fmt.Errorf("userService.GetByID(%d): %w", id, err)
  ```
- [ ] Pattern: sentinel errors for known conditions
  ```
  var ErrNotFound = errors.New("not found")
  if errors.Is(err, ErrNotFound) { ... }
  ```
- [ ] Pattern: custom error types for rich error information
  ```
  type APIError struct { Code int; Message string; Cause error }
  func (e *APIError) Error() string { ... }
  func (e *APIError) Unwrap() error { return e.Cause }
  ```
- [ ] Pattern: error classification (retryable, transient, permanent)
- [ ] Pattern: error handling middleware (HTTP, gRPC)
- [ ] Pattern: error aggregation for batch operations
- [ ] `errors.Join()` for multiple errors (Go 1.20+)
- [ ] `errgroup` for concurrent error handling with cancellation
- [ ] Pattern: error logging vs error returning (never both at the same level)
- [ ] Pattern: error handling at boundaries (convert internal errors to API errors)
- [ ] Pattern: panic recovery middleware
- [ ] Pattern: error metrics and alerting (count errors by type)
- [ ] `pkg/errors` (deprecated but still seen) vs standard library wrapping
- [ ] `hashicorp/go-multierror` for multi-error aggregation

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | CLI tools must provide clear, actionable error messages. Error classification determines retry behavior. |
| **Platform Engineering** | API error contracts between services. Error translation at service boundaries. |
| **Kubernetes** | Controllers must handle errors gracefully — requeue on transient errors, fail on permanent ones. Error conditions in status subresources. |
| **MLOps** | Training job failures need rich error context for debugging. Pipeline step errors determine retry vs fail behavior. |
| **LLM Infrastructure** | Inference errors (OOM, timeout, invalid input) need distinct handling. Error budgets for SLOs. |
| **Cloud Infrastructure** | Cloud API errors are classified by HTTP status. Retry logic depends on error type. |
| **GPU ML Systems** | CUDA errors, driver errors, memory errors — each needs specific handling and reporting. |
| **Distributed Systems** | Network errors, consensus failures, split-brain detection — error handling IS the core of distributed systems. |

### Practical Exercises

- [ ] Build a three-layer application (handler → service → repository) with proper error wrapping at each level
- [ ] Implement an error classification system: `IsRetryable(err) bool`, `IsNotFound(err) bool`, `IsPermissionDenied(err) bool`
- [ ] Build an HTTP error handling middleware that converts internal errors to JSON API error responses
- [ ] Use `errgroup` to fetch data from 5 sources concurrently; cancel all on first error
- [ ] Implement error metrics: count errors by type using a counter map
- [ ] Build a batch processor that collects all errors instead of failing on first

### Mini Projects

- [ ] **Production Error Framework**: Build a comprehensive error handling package with:
  - Error wrapping with stack traces
  - Error codes (machine-readable)
  - Error classification (retryable, not-found, permission-denied, internal)
  - JSON serialization for API responses
  - Error metrics integration
- [ ] **Resilient Service Client**: Build an HTTP service client with:
  - Automatic retry on transient errors with exponential backoff
  - Circuit breaker pattern
  - Error classification from HTTP status codes
  - Structured error logging

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Design Underground System | LeetCode #1396 | Medium | Error conditions in state management |
| LFU Cache | LeetCode #460 | Hard | Complex error conditions |
| Design In-Memory File System | LeetCode #588 | Hard | Error handling for invalid paths |
| All O'one Data Structure | LeetCode #432 | Hard | Edge case handling |

### Interview Focused Notes

**Common Interview Questions:**
- Walk me through how you handle errors in a production Go service.
- How do you decide between sentinel errors and custom error types?
- How do you handle errors in concurrent code?
- What is the error logging vs returning rule?
- How do you translate internal errors to API errors?
- How do you implement retry logic based on error types?

**Common Mistakes:**
- Logging AND returning the same error (double logging)
- Not wrapping errors (losing context)
- Wrapping with `%v` instead of `%w` (breaking the error chain)
- Catching all errors the same way (not classifying)
- Using `panic` for error handling in library code
- Not implementing `Unwrap()` on custom error types

**Interviewer Expectations:**
- Has a clear, consistent error handling strategy
- Wraps errors with context at every layer
- Classifies errors appropriately (retryable vs permanent)
- Handles errors differently at service boundaries (conversion to API errors)
- Knows `errors.Is`, `errors.As`, `errors.Join`, `errgroup`

---

## 3.4 Interface Design and Usage

### Concepts to Learn

- [ ] Interface definition: a set of method signatures
- [ ] Implicit interface implementation (no `implements` keyword)
- [ ] Empty interface: `interface{}` (any) — holds any value (Go 1.18+: `any` alias)
- [ ] "The bigger the interface, the weaker the abstraction"
- [ ] Small interfaces: `io.Reader`, `io.Writer`, `io.Closer`, `fmt.Stringer`, `error`
- [ ] Interface composition: `io.ReadWriter`, `io.ReadCloser`, `io.ReadWriteCloser`
- [ ] "Accept interfaces, return structs" principle
- [ ] Interface as contract: define behavior, not data
- [ ] Interface satisfaction checking: compile-time with `var _ Interface = (*Struct)(nil)`
- [ ] Type assertion: `val, ok := iface.(ConcreteType)`
- [ ] Type switch: `switch v := iface.(type) { ... }`
- [ ] Interface embedding in structs (for partial implementation, decoration)
- [ ] The `io.Reader` / `io.Writer` ecosystem: why these are the most important interfaces
- [ ] `sort.Interface`: `Len()`, `Less()`, `Swap()`
- [ ] `http.Handler` interface: `ServeHTTP(w, r)`
- [ ] `encoding.Marshaler` / `encoding.Unmarshaler`
- [ ] `fmt.Stringer`: `String() string`
- [ ] `context.Context` as an interface
- [ ] Interface values: (type, value) pairs; nil interfaces vs interfaces holding nil
- [ ] Interface pollution: don't define interfaces until you need them
- [ ] Consumer-defined interfaces (define where used, not where implemented)
- [ ] Mocking with interfaces for testing

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Pluggable backends (different cloud providers behind same interface). Testable code through interfaces. |
| **Platform Engineering** | Service abstractions, pluggable storage backends, extensible middleware. Interfaces define the platform contract. |
| **Kubernetes** | Kubernetes uses interfaces extensively: `runtime.Object`, `client.Client`, `reconcile.Reconciler`. Understanding these is essential for extending Kubernetes. |
| **MLOps** | Model store interface (S3, GCS, local), training backend interface (local, Kubernetes, cloud). |
| **LLM Infrastructure** | Model backend interface (vLLM, Triton, local), embedding provider interface, vector store interface. |
| **Cloud Infrastructure** | Cloud provider interface (abstract away AWS/GCP/Azure), storage interface, compute interface. |
| **GPU ML Systems** | Device manager interface, scheduler interface, resource allocator interface. |
| **Distributed Systems** | Transport interface (TCP, UDP, QUIC), consensus interface, storage engine interface. |

### Practical Exercises

- [ ] Define a `Storage` interface with `Get`, `Put`, `Delete`, `List` methods
- [ ] Implement `Storage` for in-memory, file-system, and mock (for tests)
- [ ] Use compile-time interface satisfaction checks
- [ ] Compose small interfaces: `Reader`, `Writer` → `ReadWriter`
- [ ] Implement `fmt.Stringer` for a custom type
- [ ] Implement `sort.Interface` for a custom collection
- [ ] Write a function that accepts `io.Reader` and works with files, strings, HTTP bodies, etc.
- [ ] Define a consumer-side interface in the package that uses it (not in the package that implements it)
- [ ] Write tests using interface mocks (no external mocking library, just interface implementations)

### Mini Projects

- [ ] **Plugin System**: Build a plugin system with interfaces: `type Plugin interface { Name() string; Init(config map[string]string) error; Execute(ctx context.Context) error }`. Load plugins dynamically.
- [ ] **Multi-Backend Storage**: Implement a `BlobStore` interface with S3, GCS, and local filesystem backends. Write one set of tests that validates any backend.
- [ ] **Notification System**: Interface-based notification system with Email, Slack, Webhook implementations and a composite notifier that sends to all.

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Design Iterator | LeetCode #284 | Medium | Interface-based iterator pattern |
| Flatten Nested List Iterator | LeetCode #341 | Medium | Interface with nested types |
| Design Linked List | LeetCode #707 | Medium | Interface methods on data structure |
| Insert Delete GetRandom O(1) | LeetCode #380 | Medium | Interface-worthy data structure |
| Snapshot Array | LeetCode #1146 | Medium | Versioned interface |

### Interview Focused Notes

**Common Interview Questions:**
- How do interfaces work in Go? (Implicit satisfaction)
- What does "accept interfaces, return structs" mean?
- Why are small interfaces preferred in Go?
- What is the io.Reader interface and why is it important?
- How do you test code that depends on external services? (Interfaces + mocks)
- What is the nil interface gotcha?
- What is interface pollution?
- Where should interfaces be defined — with the implementer or the consumer?

**Common Mistakes:**
- Defining interfaces prematurely (before there are multiple implementations)
- Defining interfaces where implemented instead of where consumed
- Making interfaces too large (more than 3-5 methods is a code smell)
- Using `interface{}` / `any` too freely (losing type safety)
- Not understanding implicit satisfaction (thinking you need explicit declaration)
- Embedding interfaces in structs without understanding the implications (nil panic on unimplemented methods)

**Interviewer Expectations:**
- Can design clean, minimal interfaces
- Follows "accept interfaces, return structs"
- Uses interfaces for testability (dependency injection)
- Knows the key standard library interfaces
- Can explain interface internals (type, value pair)

---

## 3.5 Composition Over Inheritance

### Concepts to Learn

- [ ] Go has NO inheritance — only composition
- [ ] Struct embedding: embedding one struct in another
- [ ] Field promotion: embedded struct fields and methods are promoted
- [ ] Method set of embedded types: how method sets compose
- [ ] Embedding interfaces in structs: partial implementation pattern
- [ ] Embedding interfaces in interfaces: interface composition
- [ ] "Has-a" relationships (composition) vs "is-a" relationships (inheritance)
- [ ] Decorator pattern using embedding
- [ ] Middleware pattern using composition
- [ ] Strategy pattern using interfaces and composition
- [ ] The delegation pattern: embedding with method overriding (shadowing)
- [ ] When to embed vs when to use a named field
  - Embed: when the outer type "is" the inner type (promotes methods)
  - Named field: when the outer type "has" the inner type (explicit access)
- [ ] Composition in real Kubernetes code: `metav1.TypeMeta`, `metav1.ObjectMeta` embedded in every resource

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Building extensible tools using composition. Plugin systems, middleware chains. |
| **Platform Engineering** | Platform abstractions built through composition. Shared base behaviors embedded in specialized types. |
| **Kubernetes** | Every Kubernetes resource type embeds `TypeMeta` and `ObjectMeta`. Understanding this is mandatory for CRD development. |
| **MLOps** | Composable pipeline steps, shared base configs embedded in specialized configs. |
| **LLM Infrastructure** | Composable middleware (auth, logging, rate limiting), shared request handling embedded in specialized handlers. |
| **Cloud Infrastructure** | Resource types composing shared metadata, provider-specific types embedding common fields. |
| **GPU ML Systems** | Device types embedding common device info, specialized GPU configs embedding base configs. |
| **Distributed Systems** | Protocol messages embedding common headers, node types embedding common state. |

### Practical Exercises

- [ ] Create a `BaseResource` with common fields (ID, CreatedAt, UpdatedAt); embed it in `User`, `Project`, `Task`
- [ ] Implement the decorator pattern: `LoggingStorage` that embeds `Storage` interface, logs each call, and delegates
- [ ] Implement HTTP middleware using composition: `AuthMiddleware(next http.Handler) http.Handler`
- [ ] Show method shadowing: embed a type, "override" one method, call the original via the embedded field
- [ ] Build a Kubernetes-style resource: embed `TypeMeta` and `ObjectMeta`, add Spec and Status

### Mini Projects

- [ ] **Composable HTTP Server**: Build an HTTP server where handlers are composed through middleware chains, each middleware is a decorator that embeds the next handler
- [ ] **Kubernetes-Style API Objects**: Implement a mini Kubernetes-like type system with `TypeMeta`, `ObjectMeta`, `Spec`, `Status` using composition

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Design Twitter | LeetCode #355 | Medium | Composition of data structures |
| Implement Trie | LeetCode #208 | Medium | Composed node structure |
| Design Log Storage System | LeetCode #635 | Medium | Composed time-based storage |
| Design Search Autocomplete System | LeetCode #642 | Hard | Trie + priority queue composition |

### Interview Focused Notes

**Common Interview Questions:**
- How does Go achieve code reuse without inheritance?
- What is struct embedding and how does method promotion work?
- When would you use embedding vs a named field?
- How do you implement the decorator pattern in Go?
- How does Kubernetes use composition in its type system?
- What are the gotchas of struct embedding?

**Common Mistakes:**
- Using embedding to "inherit" when a named field is more appropriate
- Not understanding that embedded methods can be "overridden" (shadowed)
- Embedding a mutex (exposes Lock/Unlock as exported methods if struct is exported)
- Confusion about method sets when embedding pointer vs value types
- Thinking embedding creates a "subclass" — it doesn't

**Interviewer Expectations:**
- Understands composition is Go's ONLY code reuse mechanism
- Can demonstrate multiple composition patterns (decorator, middleware, delegation)
- Knows when to embed vs use named fields
- Can design clean type hierarchies using composition

---

---

---

# STAGE 4: OBJECT-ORIENTED DESIGN IN GO

---

## 4.1 Struct-Based Design and Domain Modeling

### Concepts to Learn

- [ ] Modeling real-world entities as Go structs
- [ ] Value objects vs entity objects
- [ ] Struct as the primary unit of abstraction in Go
- [ ] Method sets: which methods belong to a type
- [ ] Value receiver methods vs pointer receiver methods (choosing correctly)
- [ ] Constructor functions: `func NewX(...) *X` pattern
- [ ] Functional options pattern: `func WithTimeout(d time.Duration) Option`
- [ ] Builder pattern in Go (less common, but used for complex objects)
- [ ] Factory pattern: returning interface implementations based on input
- [ ] Encapsulation through unexported fields and exported methods
- [ ] Validation in constructors: enforcing invariants at creation time
- [ ] Immutable structs: using unexported fields with getter methods
- [ ] Mutable state management: pointer receivers for state mutation
- [ ] Struct lifecycle: creation → use → cleanup; implementing `Close()` or `Shutdown()`
- [ ] The `Option` pattern for optional parameters:
  ```go
  type Option func(*Config)
  func WithPort(p int) Option { return func(c *Config) { c.Port = p } }
  func New(opts ...Option) *Server { ... }
  ```

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | CLI tool configuration objects, deployment target models, pipeline step definitions. |
| **Platform Engineering** | Tenant models, resource quota objects, service catalog entries. Domain modeling IS platform engineering. |
| **Kubernetes** | Every CRD is domain modeling. Spec/Status split is a design pattern. Understanding struct design is essential for controllers. |
| **MLOps** | Model metadata, training job specs, experiment tracking records, pipeline definitions. |
| **LLM Infrastructure** | Inference request models, model registry entries, serving configuration, prompt templates. |
| **Cloud Infrastructure** | Cloud resource representations, IaC resource definitions, provider configuration. |
| **GPU ML Systems** | GPU device models, memory allocation records, training cluster configurations. |
| **Distributed Systems** | Protocol message types, node state models, cluster configuration, consensus state machines. |

### Practical Exercises

- [ ] Design a `Deployment` struct with Spec and Status, using the functional options pattern for construction
- [ ] Implement a `User` entity with validation in the constructor (email format, name length, etc.)
- [ ] Build a factory function that returns different `Notifier` implementations (email, slack, webhook) based on config
- [ ] Create an immutable `Credentials` struct with getters but no setters
- [ ] Implement the builder pattern for a complex `Query` object
- [ ] Design a struct with a `Close()` method and use `defer` for cleanup

### Mini Projects

- [ ] **Domain Model Library for a Platform**: Design the complete domain model for a simple PaaS:
  - `Tenant`, `Project`, `Application`, `Deployment`, `Environment`, `Secret`, `ConfigMap`
  - Each with proper constructors, validation, and methods
  - Demonstrate relationships (tenant has many projects, project has many applications)
- [ ] **Configuration Management System**: Build a config system using functional options:
  - `Server`, `Database`, `Cache`, `Logger` configs
  - Each composable with options
  - Hierarchical defaults (env → file → defaults)

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Design Parking System | LeetCode #1603 | Easy | Struct state management |
| Design HashMap | LeetCode #706 | Easy | Struct-based data structure |
| Design Linked List | LeetCode #707 | Medium | Method-based data structure |
| LRU Cache | LeetCode #146 | Medium | Complex struct with multiple methods |
| Design Twitter | LeetCode #355 | Medium | Multi-struct system design |
| Design In-Memory File System | LeetCode #588 | Hard | Complex struct hierarchy |

### Interview Focused Notes

**Common Interview Questions:**
- How do you achieve encapsulation in Go?
- What is the functional options pattern? When would you use it?
- How do you handle required vs optional parameters in Go constructors?
- What is the difference between the builder pattern and functional options?
- How do you enforce invariants in Go (validation)?
- What is the Spec/Status pattern in Kubernetes?

**Common Mistakes:**
- Exported fields when they should be private (breaking encapsulation)
- No validation in constructors (invalid objects in the wild)
- Mixing value and pointer receivers on the same type
- Over-engineering with too many patterns (keep it simple)
- Not providing a `Close()`/`Shutdown()` method for types that hold resources

**Interviewer Expectations:**
- Can design clean domain models with proper encapsulation
- Uses functional options for complex constructors
- Validates input at construction time
- Designs types with clear lifecycle (create → use → cleanup)

---

## 4.2 Dependency Injection in Go

### Concepts to Learn

- [ ] What is dependency injection (DI): passing dependencies rather than creating them
- [ ] Constructor injection: pass dependencies as constructor parameters
- [ ] Interface-based DI: depend on interfaces, inject implementations
- [ ] The DI principle: "Depend on abstractions, not on concretions"
- [ ] Wire (Google's DI tool): code generation for DI
- [ ] `fx` (Uber's DI framework): runtime DI with lifecycle management
- [ ] Manual DI vs framework-based DI (prefer manual in Go)
- [ ] DI for testability: inject mocks in tests
- [ ] Avoiding global state: pass dependencies explicitly
- [ ] Service struct pattern: struct that holds dependencies and exposes methods
  ```go
  type UserService struct {
      repo   UserRepository  // interface
      cache  Cache           // interface
      logger *zap.Logger
  }
  func NewUserService(repo UserRepository, cache Cache, logger *zap.Logger) *UserService { ... }
  ```
- [ ] DI in HTTP handlers: injecting services into handlers
- [ ] DI in Kubernetes controllers: injecting clients, caches, loggers

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Testable CLI tools with injected dependencies (cloud client, git client, file system). |
| **Platform Engineering** | Service layer with injected storage, cache, message queue. Platform services are DI-heavy. |
| **Kubernetes** | Controllers inject client, cache, recorder, logger. Kubebuilder wires these automatically. |
| **MLOps** | Pipeline runners with injected compute backend, storage backend, metrics collector. |
| **LLM Infrastructure** | Inference services with injected model loader, tokenizer, cache, rate limiter. |
| **Cloud Infrastructure** | Cloud clients, storage backends, and monitoring injected into infrastructure tools. |
| **GPU ML Systems** | Device managers, schedulers, monitoring injected into GPU management services. |
| **Distributed Systems** | Transport layer, storage engine, consensus module all injected into node struct. |

### Practical Exercises

- [ ] Refactor a service that directly creates its dependencies to use constructor injection
- [ ] Write unit tests for a service using mock implementations of injected interfaces
- [ ] Build a simple HTTP server with DI: handler → service → repository → database
- [ ] Set up `google/wire` for a small project and see the generated code
- [ ] Implement the service struct pattern for a CRUD service

### Mini Projects

- [ ] **Clean Architecture Microservice**: Build a microservice with strict layers:
  - `domain/` — entities and interfaces (no external dependencies)
  - `usecase/` — business logic, depends only on domain interfaces
  - `adapter/` — implementations (PostgreSQL repo, Redis cache, HTTP handler)
  - `cmd/` — wiring (DI, creating all dependencies and injecting them)
- [ ] **Testable Kubernetes Controller**: Build a controller where the reconciler has injected dependencies (Kubernetes client, external API client, metrics recorder) — all testable with mocks

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Design Tic-Tac-Toe | LeetCode #348 | Medium | Strategy injection |
| Design Hit Counter | LeetCode #362 | Medium | Storage abstraction |
| Design Snake Game | LeetCode #353 | Medium | Composed dependencies |
| Design File System | LeetCode #1166 | Medium | Layer dependencies |

### Interview Focused Notes

**Common Interview Questions:**
- How do you do dependency injection in Go?
- Do you use a DI framework? Why or why not?
- How does DI improve testability?
- Walk me through how you structure a production Go service with DI.
- How do you inject dependencies into HTTP handlers?
- What is the difference between constructor injection and method injection?

**Common Mistakes:**
- Using global variables instead of DI (makes testing impossible)
- Over-using DI frameworks (manual DI is fine for most Go projects)
- Injecting concrete types instead of interfaces
- Creating a "God struct" that holds too many dependencies (split into focused services)
- Not providing defaults for optional dependencies

**Interviewer Expectations:**
- Uses interfaces for dependency abstraction
- Injects dependencies through constructors
- Can write testable code with mock dependencies
- Knows when to use manual DI vs a framework
- Clean separation of concerns across layers

---

## 4.3 Clean Architecture and Domain-Driven Design

### Concepts to Learn

- [ ] Clean Architecture layers:
  - **Entities (Domain)**: core business rules, no external dependencies
  - **Use Cases (Application)**: application-specific business rules
  - **Interface Adapters**: controllers, presenters, gateways
  - **Frameworks & Drivers**: database, web, UI, external services
- [ ] Dependency rule: dependencies point inward (outer layers depend on inner)
- [ ] Domain-Driven Design (DDD) concepts:
  - **Entity**: object with identity (User, Order)
  - **Value Object**: object defined by its attributes (Money, Address)
  - **Aggregate**: cluster of entities treated as a unit (Order + OrderItems)
  - **Aggregate Root**: entry point to the aggregate
  - **Repository**: interface for data access
  - **Service**: domain logic that doesn't belong to an entity
  - **Domain Event**: something that happened in the domain
- [ ] Hexagonal Architecture (Ports and Adapters):
  - **Port**: interface defining how the application interacts with the outside
  - **Adapter**: implementation of a port (PostgreSQL adapter, HTTP adapter)
- [ ] CQRS (Command Query Responsibility Segregation): separate read and write models
- [ ] Event Sourcing: storing events instead of current state
- [ ] Repository pattern in Go: interface in domain, implementation in adapter
- [ ] Use case / service pattern in Go
- [ ] Anti-corruption layer: translating between domains

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Large automation platforms benefit from clean architecture. Separating domain logic from infrastructure details. |
| **Platform Engineering** | Platform services are complex — clean architecture keeps them maintainable. Multi-tenant platforms use DDD aggregates. |
| **Kubernetes** | Kubernetes controllers follow a pattern similar to hexagonal architecture: reconciler (use case) depends on interfaces (ports), concrete clients (adapters). |
| **MLOps** | ML platforms with complex domain models (experiments, runs, metrics) benefit from DDD. Repository pattern for model/artifact storage. |
| **LLM Infrastructure** | LLM platforms model prompts, conversations, and deployments as domain entities. Clean architecture separates inference logic from serving infrastructure. |
| **Cloud Infrastructure** | Multi-cloud abstractions use hexagonal architecture. Cloud-specific implementations behind common interfaces. |
| **GPU ML Systems** | Resource scheduling domain modeled as entities and aggregates. Clean separation between scheduling logic and hardware interaction. |
| **Distributed Systems** | State machines, consensus protocols, and membership management modeled as domain entities with clear boundaries. |

### Practical Exercises

- [ ] Design a clean architecture project structure for a user management service
- [ ] Implement the repository pattern: `UserRepository` interface in domain, PostgreSQL implementation in adapter
- [ ] Build a use case that orchestrates multiple repositories and services
- [ ] Implement a domain event system: define events, publish them, handle them
- [ ] Create a value object `Money` with currency and amount, proper equality
- [ ] Implement an aggregate (Order + OrderItems) with invariant enforcement

### Mini Projects

- [ ] **Clean Architecture API**: Build a complete REST API following clean architecture:
  - Domain layer: entities, value objects, repository interfaces
  - Use case layer: business logic, input/output ports
  - Adapter layer: HTTP handlers, PostgreSQL repos, Redis cache
  - Wire everything in `cmd/main.go`
  - Full test coverage at each layer
- [ ] **Event-Sourced Task Manager**: Build a task management system using event sourcing:
  - Events: TaskCreated, TaskAssigned, TaskCompleted, TaskCommented
  - Event store (in-memory or PostgreSQL)
  - Projections for read models
  - Snapshots for performance

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Design Underground System | LeetCode #1396 | Medium | Domain modeling |
| Time Based Key-Value Store | LeetCode #981 | Medium | Repository-like pattern |
| Design Movie Rental System | LeetCode #1912 | Hard | Complex domain with aggregates |
| Stock Price Fluctuation | LeetCode #2034 | Medium | CQRS-like read/write patterns |

### Interview Focused Notes

**Common Interview Questions:**
- What is clean architecture and how do you implement it in Go?
- What is the repository pattern?
- What is the difference between an entity and a value object?
- What is hexagonal architecture?
- How do you keep your domain layer free of external dependencies?
- What is CQRS and when would you use it?
- How do you test each layer independently?

**Common Mistakes:**
- Over-engineering small projects with too many layers
- Leaking infrastructure details into the domain layer (SQL queries in entities)
- Making the domain depend on the database (wrong dependency direction)
- Not defining clear aggregate boundaries (too large or too small)
- Using DDD vocabulary without understanding the concepts

**Interviewer Expectations:**
- Can explain clean architecture layers and dependency rules
- Knows when clean architecture is worth the complexity
- Can implement the repository pattern properly
- Understands DDD concepts and can apply them pragmatically
- Clean separation of concerns in code organization

---

---

# STAGE 5: ADVANCED GO — Concurrency and the Runtime

---

## 5.1 Goroutines

### Concepts to Learn

- [ ] What is a goroutine: lightweight thread managed by Go runtime
- [ ] Creating a goroutine: `go func() { ... }()`
- [ ] Goroutine vs OS thread: goroutines are multiplexed onto OS threads
- [ ] Goroutine stack: starts small (~2KB), grows/shrinks dynamically (up to 1GB default)
- [ ] Goroutine lifecycle: creation, execution, blocking, completion
- [ ] Goroutine scheduling: cooperative scheduling, preemption points
- [ ] `runtime.GOMAXPROCS()`: controls the number of OS threads used
- [ ] `runtime.NumGoroutine()`: how many goroutines are running
- [ ] `runtime.Gosched()`: yield the processor (rarely needed)
- [ ] `runtime.Goexit()`: terminate the goroutine (runs defers, unlike `os.Exit`)
- [ ] Goroutine leaks: goroutines that never terminate (common production bug)
- [ ] Detecting goroutine leaks: `runtime.NumGoroutine()`, `goleak` library
- [ ] Common causes of goroutine leaks:
  - Blocked on channel with no sender/receiver
  - Blocked on mutex that's never unlocked
  - Infinite loop without exit condition
  - Network call without timeout
- [ ] Goroutine patterns:
  - Fire and forget (dangerous, avoid in production)
  - With synchronization (WaitGroup, channel, context)
  - With error handling (errgroup)
- [ ] Goroutine naming and identification (no built-in names, use goroutine ID from stack for debugging)

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Concurrent task execution (parallel deployments, multi-server health checks, concurrent log collection). |
| **Platform Engineering** | Platform services handle thousands of concurrent requests. Understanding goroutines is fundamental. |
| **Kubernetes** | Controllers run reconciliation loops in goroutines. Informers run in goroutines. Work queues process items with goroutine workers. |
| **MLOps** | Parallel data preprocessing, concurrent model evaluation, parallel hyperparameter search. |
| **LLM Infrastructure** | Concurrent inference requests, parallel token generation, concurrent embedding computation. |
| **Cloud Infrastructure** | Concurrent API calls to cloud providers, parallel resource provisioning, concurrent health checks. |
| **GPU ML Systems** | Concurrent GPU device monitoring, parallel job scheduling, concurrent metrics collection. |
| **Distributed Systems** | Every node runs many goroutines: request handling, heartbeats, consensus, replication, compaction. |

### Practical Exercises

- [ ] Launch 10 goroutines that print their IDs; observe non-deterministic ordering
- [ ] Create a goroutine leak: launch a goroutine blocked on a channel, observe `NumGoroutine` growing
- [ ] Fix the goroutine leak using context cancellation
- [ ] Launch 1000 goroutines and measure memory usage (demonstrate lightweight nature)
- [ ] Use `runtime.GOMAXPROCS(1)` to force single-threaded execution and observe behavior
- [ ] Create a program that demonstrates cooperative scheduling (goroutine that never yields blocks others on single thread)

### Mini Projects

- [ ] **Concurrent Health Checker**: Check health of 100 HTTP endpoints concurrently with goroutines, collect results, and display a summary
- [ ] **Goroutine Pool**: Implement a simple goroutine pool that limits the number of concurrent goroutines (using a buffered channel as a semaphore)
- [ ] **Leak Detector**: Build a test helper that checks `runtime.NumGoroutine()` before and after a test to detect goroutine leaks

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Print in Order | LeetCode #1114 | Easy | Goroutine ordering |
| Print FooBar Alternately | LeetCode #1115 | Medium | Goroutine synchronization |
| Building H2O | LeetCode #1117 | Medium | Goroutine coordination |
| The Dining Philosophers | LeetCode #1226 | Medium | Classic concurrency problem |
| Fizz Buzz Multithreaded | LeetCode #1195 | Medium | Concurrent output |

### Interview Focused Notes

**Common Interview Questions:**
- What is a goroutine and how does it differ from a thread?
- How many goroutines can you run? (Millions — limited by memory)
- What is a goroutine leak? How do you detect and prevent it?
- How does the Go scheduler work? What is M:N scheduling?
- What is `GOMAXPROCS` and what does it control?
- How do you wait for goroutines to finish?

**Common Mistakes:**
- Launching goroutines without any way to wait for them
- Not using context for cancellation (goroutine leaks)
- Accessing shared state from goroutines without synchronization (data race)
- Launching goroutines in a loop with closure variable capture bug
- Not handling panics in goroutines (kills the entire program)

**Interviewer Expectations:**
- Can explain goroutine internals (stack, scheduling, M:N model)
- Knows goroutine leak patterns and prevention
- Uses synchronization correctly (channels, WaitGroup, context)
- Can reason about concurrent program behavior

---

## 5.2 Channels

### Concepts to Learn

- [ ] Channel concept: typed conduit for goroutine communication
- [ ] Creating channels: `ch := make(chan int)` (unbuffered), `ch := make(chan int, 10)` (buffered)
- [ ] Send: `ch <- value` (blocks if unbuffered and no receiver, or buffer full)
- [ ] Receive: `value := <-ch` (blocks if no sender and buffer empty)
- [ ] Closing channels: `close(ch)` — receive returns zero value, send panics
- [ ] Range over channel: `for val := range ch { ... }` — loops until channel is closed
- [ ] Comma-ok pattern: `val, ok := <-ch` — `ok` is false if channel is closed
- [ ] Channel direction: `chan<-` (send-only), `<-chan` (receive-only)
- [ ] Unbuffered vs buffered channels:
  - Unbuffered: synchronous, sender blocks until receiver is ready (rendezvous)
  - Buffered: asynchronous up to buffer size, sender blocks only when full
- [ ] Channel as signaling mechanism: `done := make(chan struct{})`
- [ ] Channel axioms:
  - Send to nil channel: blocks forever
  - Receive from nil channel: blocks forever
  - Send to closed channel: panic
  - Receive from closed channel: returns zero value immediately
  - Close nil channel: panic
  - Close closed channel: panic
- [ ] Channel patterns:
  - Done channel for cancellation
  - Pipeline pattern (chain of processing stages)
  - Fan-out (multiple goroutines reading from one channel)
  - Fan-in (multiple channels merged into one)
  - Or-channel (first result wins)
  - Tee channel (duplicate a channel)
  - Bridge channel (flatten channel of channels)
  - Semaphore (buffered channel for concurrency limiting)
- [ ] Channel vs mutex: when to use which
  - Channel: transferring ownership of data, coordinating goroutines
  - Mutex: protecting shared state

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Pipeline processing (data flows through stages), concurrent task coordination, result aggregation. |
| **Platform Engineering** | Event streaming, work distribution, rate limiting with buffered channels. |
| **Kubernetes** | Work queues internally use channels. Informer event handlers push to channels. Controller reconciliation is channel-based. |
| **MLOps** | Data pipeline stages connected by channels, training metric streaming, concurrent data loading. |
| **LLM Infrastructure** | Token streaming (channel per request), batch aggregation, concurrent request handling. |
| **Cloud Infrastructure** | Concurrent API pagination, parallel resource discovery, event streaming. |
| **GPU ML Systems** | GPU job queuing, device event channels, training progress streaming. |
| **Distributed Systems** | Message passing between components, event buses, internal pub-sub systems. |

### Practical Exercises

- [ ] Create an unbuffered channel; demonstrate that send blocks until receive
- [ ] Create a buffered channel; demonstrate that send doesn't block until buffer is full
- [ ] Implement a pipeline: generate → square → print, each stage in its own goroutine
- [ ] Implement fan-out: one producer, three consumers reading from the same channel
- [ ] Implement fan-in: three producers writing to separate channels, one consumer merging
- [ ] Demonstrate all channel axioms (nil channel, closed channel behaviors)
- [ ] Implement a semaphore using a buffered channel to limit to 5 concurrent goroutines
- [ ] Use `range` over a channel with proper closing
- [ ] Build an or-channel that returns the result from whichever goroutine finishes first

### Mini Projects

- [ ] **Data Processing Pipeline**: Build a multi-stage pipeline:
  - Stage 1: Read lines from a file (channel of strings)
  - Stage 2: Parse each line into a struct (channel of structs)
  - Stage 3: Transform/enrich data (channel of enriched structs)
  - Stage 4: Write to output file
  - Each stage runs in goroutines, connected by channels
- [ ] **Job Queue System**: Build an in-memory job queue with:
  - Producer goroutines submitting jobs to a channel
  - Worker pool consuming from the channel
  - Result channel for collecting outcomes
  - Graceful shutdown via done channel
- [ ] **Event Bus**: Build an in-process event bus:
  - `Subscribe(topic string) <-chan Event`
  - `Publish(topic string, event Event)`
  - `Unsubscribe(topic string, ch <-chan Event)`

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Print in Order | LeetCode #1114 | Easy | Channel synchronization |
| Print FooBar Alternately | LeetCode #1115 | Medium | Alternating channels |
| Print Zero Even Odd | LeetCode #1116 | Medium | Multi-channel coordination |
| Web Crawler Multithreaded | LeetCode #1242 | Medium | Channel-based BFS |
| Design Bounded Blocking Queue | LeetCode #1188 | Medium | Buffered channel semantics |

### Interview Focused Notes

**Common Interview Questions:**
- What is the difference between buffered and unbuffered channels?
- What happens when you send to a closed channel? Receive from a closed channel?
- When would you use a channel vs a mutex?
- Explain the pipeline pattern with channels.
- What is fan-out/fan-in? When would you use it?
- How do you prevent goroutine leaks with channels?
- What is a nil channel useful for? (Disabling a case in select)

**Common Mistakes:**
- Sending to a closed channel (panic)
- Closing a channel from the receiving side (only the sender should close)
- Closing a channel multiple times (panic)
- Not closing channels (goroutine leaks in range loops)
- Using unbuffered channels when buffered is needed (deadlock)
- Forgetting that channel direction types restrict operations

**Interviewer Expectations:**
- Can whiteboard channel axioms from memory
- Knows multiple channel patterns (pipeline, fan-out, fan-in, semaphore)
- Can reason about deadlock scenarios
- Chooses correctly between channels and mutexes
- Can implement a producer-consumer system with channels

---

## 5.3 Select Statement

### Concepts to Learn

- [ ] `select` statement: multiplexes channel operations
- [ ] Syntax: `select { case v := <-ch1: ... case ch2 <- val: ... default: ... }`
- [ ] Blocking select: waits until one case is ready (no default)
- [ ] Non-blocking select: `default` case makes it non-blocking
- [ ] Random selection: if multiple cases are ready, one is chosen randomly
- [ ] `select {}`: blocks forever (useful for keeping main goroutine alive)
- [ ] Timeout with select: `case <-time.After(5 * time.Second):`
- [ ] Context cancellation with select: `case <-ctx.Done():`
- [ ] Nil channel in select: disables that case (useful for dynamic enable/disable)
- [ ] Select with done channel for graceful shutdown
- [ ] Select in loops: the "for-select" pattern
- [ ] Priority select: Go doesn't have built-in priority, workaround with nested selects

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Timeout handling in automation scripts, graceful shutdown of long-running tasks. |
| **Platform Engineering** | Multiplexing events from multiple sources, timeout handling, health check polling. |
| **Kubernetes** | Controller reconcile loops use for-select. Informers use select for event multiplexing. Work queue processing uses select with context. |
| **MLOps** | Training loop with select for cancellation, metric reporting timeout, checkpoint trigger. |
| **LLM Infrastructure** | Request timeout handling, streaming response with cancellation, load balancer with health check polling. |
| **Cloud Infrastructure** | Polling cloud APIs with timeout, event-driven resource management. |
| **GPU ML Systems** | GPU event polling, device health monitoring with timeout, job cancellation. |
| **Distributed Systems** | Heartbeat sending with timeout, leader election with select, message routing with multiplexing. |

### Practical Exercises

- [ ] Write a select that reads from two channels and a timeout
- [ ] Implement a non-blocking channel read with `default`
- [ ] Build a heartbeat system: send heartbeats on a channel, detect missed heartbeats with timeout
- [ ] Implement graceful shutdown: listen on `os.Signal` channel AND a done channel
- [ ] Use nil channels to dynamically enable/disable select cases
- [ ] Build a priority select that prefers one channel over another
- [ ] Implement a ticker with select that stops on context cancellation

### Mini Projects

- [ ] **Graceful Shutdown Manager**: Build a shutdown manager that:
  - Listens for OS signals (SIGTERM, SIGINT)
  - Signals all goroutines to stop via context
  - Waits for goroutines to finish with a timeout
  - Force-kills after timeout
- [ ] **Multi-Source Event Aggregator**: Read events from 5 different channels (different event sources), aggregate them, and process with timeout handling for slow sources

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Traffic Light Controlled Intersection | LeetCode #1279 | Easy | State machine with events |
| Design Bounded Blocking Queue | LeetCode #1188 | Medium | Blocking operations with timeout |

### Interview Focused Notes

**Common Interview Questions:**
- How does `select` work in Go?
- What happens if multiple cases are ready in a select?
- How do you implement a timeout with select?
- How do you do a non-blocking channel operation?
- What is a nil channel in select used for?
- How do you implement graceful shutdown with select?
- What is the for-select pattern?

**Common Mistakes:**
- Forgetting the `default` case when non-blocking is intended
- `time.After` in a loop creating timers every iteration (memory leak) — use `time.NewTimer` with Reset
- Not handling `ctx.Done()` in for-select loops (goroutine leak on cancellation)
- Assuming priority in select (it's random when multiple cases are ready)

**Interviewer Expectations:**
- Can explain select semantics precisely
- Knows the for-select pattern for long-running goroutines
- Can implement timeout and cancellation patterns
- Understands the timer/ticker memory leak issue

---

## 5.4 Concurrency Patterns

### Concepts to Learn

- [ ] Generator pattern: function that returns a channel
- [ ] Pipeline pattern: chain of stages connected by channels
- [ ] Fan-out / Fan-in: distribute work, collect results
- [ ] Or-Done channel: read from a channel until done
- [ ] Tee channel: split one channel into two
- [ ] Bridge channel: flatten a channel of channels
- [ ] Worker pool: fixed number of goroutines processing from a shared channel
- [ ] Bounded parallelism: limit concurrent operations with semaphore
- [ ] Rate limiting: token bucket, leaky bucket using channels and tickers
- [ ] Circuit breaker: prevent cascading failures
- [ ] Pub-Sub pattern: publishers and subscribers connected by channels
- [ ] Request-Response pattern: send request on one channel, receive response on another
- [ ] Scatter-Gather: send request to multiple backends, gather responses
- [ ] MapReduce in Go: parallel map phase, reduction phase
- [ ] Confinement: limiting goroutine access to data (lexical confinement)
- [ ] Error propagation in concurrent code
- [ ] Context propagation through goroutine chains

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Worker pools for parallel task execution, pipelines for CI/CD stages, rate limiting for API calls. |
| **Platform Engineering** | Request fan-out to multiple backends, scatter-gather for aggregation, circuit breakers for resilience. |
| **Kubernetes** | Controller work queues use worker pool pattern. Informer event handlers use fan-out. |
| **MLOps** | MapReduce for distributed training, pipeline for data preprocessing, worker pool for batch inference. |
| **LLM Infrastructure** | Fan-out to multiple model replicas, rate limiting for API endpoints, circuit breaker for model backends. |
| **Cloud Infrastructure** | Parallel resource provisioning (worker pool), multi-region deployment (scatter-gather). |
| **GPU ML Systems** | Worker pool for GPU job scheduling, pipeline for data loading → preprocessing → training. |
| **Distributed Systems** | All patterns apply directly: pub-sub, scatter-gather, circuit breaker, rate limiting. |

### Practical Exercises

- [ ] Implement a generator that produces Fibonacci numbers on a channel
- [ ] Build a 3-stage pipeline: generate numbers → filter primes → format output
- [ ] Implement fan-out/fan-in for concurrent URL fetching with 10 workers
- [ ] Build a worker pool with configurable size that processes jobs from a queue
- [ ] Implement a token bucket rate limiter using channels and tickers
- [ ] Build a circuit breaker that opens after 5 consecutive failures
- [ ] Implement scatter-gather: send a query to 3 backends, return the first successful result
- [ ] Build a MapReduce that counts word frequencies across multiple files in parallel

### Mini Projects

- [ ] **Production Worker Pool Library**: Build a reusable worker pool with:
  - Configurable pool size
  - Graceful shutdown with context
  - Error handling per job
  - Result collection
  - Metrics (jobs processed, errors, queue depth)
  - Dead letter queue for failed jobs
- [ ] **Resilience Library**: Build a Go resilience library with:
  - Circuit breaker (half-open, open, closed states)
  - Retry with exponential backoff and jitter
  - Rate limiter (token bucket)
  - Timeout wrapper
  - Bulkhead (concurrent call limiter)
  - All composable: `Retry(CircuitBreaker(RateLimit(fn)))`

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Web Crawler Multithreaded | LeetCode #1242 | Medium | Concurrent BFS with worker pool |
| Design Bounded Blocking Queue | LeetCode #1188 | Medium | Producer-consumer |
| The Dining Philosophers | LeetCode #1226 | Medium | Deadlock avoidance |
| Building H2O | LeetCode #1117 | Medium | Complex synchronization |
| Concurrent Sum (custom) | — | Medium | MapReduce pattern |

### Interview Focused Notes

**Common Interview Questions:**
- Explain the worker pool pattern. How would you implement it?
- What is fan-out/fan-in? Give a real-world example.
- How do you implement rate limiting in Go?
- What is a circuit breaker? How would you implement one?
- How do you propagate errors in concurrent Go code?
- What is the pipeline pattern? When would you use it?
- How do you handle context cancellation through a chain of goroutines?

**Common Mistakes:**
- Worker pool goroutines not exiting when the pool is stopped
- Not using buffered channels for results (deadlock when all workers try to send)
- Rate limiter that doesn't handle burst correctly
- Circuit breaker without a half-open state (no recovery)
- Pipeline stages not handling context cancellation (goroutine leaks)

**Interviewer Expectations:**
- Can implement worker pool, pipeline, fan-out/fan-in from scratch
- Knows resilience patterns (circuit breaker, retry, rate limit)
- Can combine patterns for real-world scenarios
- Handles errors and cancellation in concurrent code

---

## 5.5 Synchronization Primitives

### Concepts to Learn

- [ ] `sync.Mutex`: mutual exclusion lock
  - `Lock()`, `Unlock()`
  - Always defer Unlock: `mu.Lock(); defer mu.Unlock()`
  - Zero value is an unlocked mutex
- [ ] `sync.RWMutex`: reader-writer lock
  - `RLock()`, `RUnlock()` for read access (multiple concurrent readers)
  - `Lock()`, `Unlock()` for write access (exclusive)
  - Use when reads far outnumber writes
- [ ] `sync.WaitGroup`: wait for a collection of goroutines
  - `Add(n)`, `Done()`, `Wait()`
  - Must call `Add` before launching goroutine
  - `Done()` is equivalent to `Add(-1)`
- [ ] `sync.Once`: execute a function exactly once
  - `Do(func())`: thread-safe initialization
  - Common for singleton pattern, lazy initialization
- [ ] `sync.Map`: concurrent-safe map
  - `Store`, `Load`, `LoadOrStore`, `Delete`, `Range`
  - When to use: many goroutines reading/writing disjoint key sets
  - When NOT to use: most cases (regular map + mutex is usually better)
- [ ] `sync.Pool`: reusable object pool
  - `Get()`, `Put()`
  - Reduces GC pressure for frequently allocated objects
  - Objects may be garbage collected between GC cycles
  - Used in `fmt`, `encoding/json`, HTTP request buffers
- [ ] `sync.Cond`: condition variable
  - `Wait()`, `Signal()`, `Broadcast()`
  - Rarely used directly (channels are preferred)
- [ ] `atomic` package: low-level atomic operations
  - `atomic.AddInt64`, `atomic.LoadInt64`, `atomic.StoreInt64`, `atomic.CompareAndSwapInt64`
  - `atomic.Value`: store and load arbitrary values atomically
  - `atomic.Int64`, `atomic.Bool` (Go 1.19+): typed atomic values
  - When to use: simple counters, flags, single-word state
  - When NOT to use: complex state (use mutex instead)

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Concurrent access to shared state (deployment status, health check results). Atomic counters for metrics. |
| **Platform Engineering** | Shared configuration (RWMutex), service registry (sync.Map or mutex+map), connection pools (sync.Pool). |
| **Kubernetes** | Informer cache uses RWMutex. Controller shared state uses mutex. Rate limiters use atomic counters. |
| **MLOps** | Training metrics (atomic counters), shared model state (RWMutex), buffer pools for data loading. |
| **LLM Infrastructure** | Request counters (atomic), model cache (RWMutex), token buffer pool (sync.Pool). |
| **Cloud Infrastructure** | Concurrent API rate counters, shared resource state, connection pooling. |
| **GPU ML Systems** | GPU allocation state (mutex), device counters (atomic), memory buffer pools. |
| **Distributed Systems** | Shared state in consensus protocols, atomic leader flag, connection pools, concurrent data structures. |

### Practical Exercises

- [ ] Build a thread-safe counter using `sync.Mutex`
- [ ] Build the same counter using `atomic.Int64` and compare performance with benchmarks
- [ ] Implement a read-heavy cache using `sync.RWMutex`
- [ ] Use `sync.WaitGroup` to launch 100 goroutines and wait for all to complete
- [ ] Use `sync.Once` to implement a lazy-initialized database connection
- [ ] Use `sync.Pool` for buffer reuse and benchmark the difference
- [ ] Create a data race detector example: run with `go run -race` and fix the race
- [ ] Build a concurrent-safe map using `sync.RWMutex` + `map` and compare with `sync.Map`

### Mini Projects

- [ ] **Thread-Safe In-Memory Cache**: Build a cache with:
  - RWMutex for concurrent access
  - TTL expiration (background goroutine with ticker)
  - LRU eviction (when cache exceeds size limit)
  - Atomic hit/miss counters for metrics
  - sync.Pool for entry objects
- [ ] **Concurrent Data Structures Library**: Implement:
  - Thread-safe stack (mutex-based)
  - Thread-safe queue (mutex-based)
  - Lock-free stack (atomic CAS-based)
  - Concurrent linked list

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| The Dining Philosophers | LeetCode #1226 | Medium | Mutex usage and deadlock avoidance |
| Print FooBar Alternately | LeetCode #1115 | Medium | Synchronization primitives |
| Design Bounded Blocking Queue | LeetCode #1188 | Medium | Mutex + condition variable |
| Traffic Light Controlled Intersection | LeetCode #1279 | Easy | Mutex for shared state |

### Interview Focused Notes

**Common Interview Questions:**
- What is the difference between `sync.Mutex` and `sync.RWMutex`? When use which?
- What is `sync.WaitGroup`? How does it work?
- What is `sync.Once`? Give a real use case.
- When would you use `sync.Pool`?
- When would you use atomic operations vs a mutex?
- What is a data race? How do you detect and prevent them?
- When would you use `sync.Map` vs a regular map with a mutex?

**Common Mistakes:**
- Forgetting to `defer mu.Unlock()` (deadlock on error/panic)
- Copying a mutex (the copy is a new unlocked mutex — data race)
- Calling `wg.Add()` inside the goroutine instead of before (race condition)
- Using `sync.Pool` for objects that need deterministic lifecycle
- Nested locking with same mutex (deadlock)
- Not using `-race` flag during development and testing

**Interviewer Expectations:**
- Can explain each sync primitive and its use case
- Knows when to use channels vs mutexes vs atomics
- Always uses `-race` flag during testing
- Can identify and fix data races
- Understands the performance tradeoffs between primitives

---

## 5.6 Context Package

### Concepts to Learn

- [ ] `context.Context` interface: `Deadline()`, `Done()`, `Err()`, `Value()`
- [ ] `context.Background()`: root context, used in main, init, tests
- [ ] `context.TODO()`: placeholder when unsure which context to use
- [ ] `context.WithCancel(parent)`: returns ctx and cancel function
- [ ] `context.WithTimeout(parent, duration)`: auto-cancels after duration
- [ ] `context.WithDeadline(parent, time)`: auto-cancels at specific time
- [ ] `context.WithValue(parent, key, value)`: attach values (use sparingly)
- [ ] Context tree: child contexts derive from parents, cancellation propagates down
- [ ] `ctx.Done()` returns a channel that closes when context is canceled
- [ ] `ctx.Err()` returns `context.Canceled` or `context.DeadlineExceeded`
- [ ] Convention: context is always the first parameter, named `ctx`
- [ ] Do NOT store context in structs (pass explicitly)
- [ ] Context values: use only for request-scoped data (trace ID, auth info, NOT business logic)
- [ ] Custom context key types to avoid collisions: `type ctxKey struct{}`
- [ ] `context.WithCancelCause` (Go 1.20+): cancel with a reason
- [ ] `context.AfterFunc` (Go 1.21+): register callback for when context is done
- [ ] Context in HTTP: `r.Context()` provides request context
- [ ] Context in database: `db.QueryContext(ctx, ...)`
- [ ] Context in gRPC: propagated through metadata

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Timeout for CLI operations, cancellation of long-running automation tasks. |
| **Platform Engineering** | Request timeout propagation through service layers, graceful shutdown, operation deadlines. |
| **Kubernetes** | Controller context from manager, reconcile context per reconciliation, context cancellation on leader loss. |
| **MLOps** | Training job cancellation, pipeline step timeout, data loading timeout. |
| **LLM Infrastructure** | Inference timeout, streaming cancellation, request deadline propagation. |
| **Cloud Infrastructure** | API call timeout, resource provisioning deadline, operation cancellation. |
| **GPU ML Systems** | GPU operation timeout, training job cancellation, health check deadline. |
| **Distributed Systems** | RPC deadline propagation, cross-service timeout, distributed trace context. |

### Practical Exercises

- [ ] Create a context hierarchy: background → with cancel → with timeout → with value
- [ ] Cancel a parent context and observe all children's `Done()` channels close
- [ ] Use `context.WithTimeout` to cancel an HTTP request after 5 seconds
- [ ] Pass trace ID through context values across function calls
- [ ] Implement a long-running operation that checks `ctx.Done()` periodically
- [ ] Build an HTTP handler that propagates its request context to database queries
- [ ] Use `context.WithCancelCause` and retrieve the cause on cancellation

### Mini Projects

- [ ] **Request-Scoped Context Pipeline**: Build an HTTP service that:
  - Extracts request ID from header (or generates one)
  - Stores request ID in context
  - Passes context through all layers (handler → service → repository)
  - All logs include request ID from context
  - Database queries use request context for timeout
  - If client disconnects, all operations cancel
- [ ] **Distributed Timeout Propagation**: Simulate a microservice chain where:
  - Service A calls Service B calls Service C
  - Timeout set at Service A propagates through B and C
  - If B is slow, C never gets called (context expired)

### Logic Building / DSA Problems

(Context is more of a systems concept — DSA problems map loosely)

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Time Needed to Inform All Employees | LeetCode #1376 | Medium | Tree traversal with propagation (like context) |
| Course Schedule | LeetCode #207 | Medium | Dependency graph (like context tree) |

### Interview Focused Notes

**Common Interview Questions:**
- What is `context.Context` and why does it exist?
- What is the difference between `WithCancel`, `WithTimeout`, and `WithDeadline`?
- When would you use `context.WithValue`? What should you NOT store in it?
- Why should context be the first parameter of a function?
- How does context cancellation propagate through a service chain?
- What happens if you don't check `ctx.Done()` in a long-running operation?

**Common Mistakes:**
- Storing context in a struct field
- Using `context.WithValue` for business logic data (only request-scoped metadata)
- Not checking `ctx.Done()` in long-running operations (ignoring cancellation)
- Using `context.Background()` when a proper context is available
- Not calling the cancel function (resource leak)
- Using `string` as context key (collisions between packages)

**Interviewer Expectations:**
- Uses context consistently as the first parameter
- Always passes cancel context to external calls (HTTP, DB, RPC)
- Checks context cancellation in long-running operations
- Uses context values appropriately (only request-scoped metadata)
- Calls cancel functions promptly (often with defer)

---

## 5.7 Go Runtime, Scheduler, and Memory

### Concepts to Learn

**Go Scheduler:**
- [ ] GMP model: Goroutines (G), OS Threads (M), Processors (P)
- [ ] P (logical processor): each P has a local run queue of goroutines
- [ ] Global run queue: overflow goroutines when local queues are full
- [ ] Work stealing: idle P steals goroutines from busy P's queue
- [ ] `GOMAXPROCS`: number of P's (default = number of CPU cores)
- [ ] Cooperative scheduling: goroutines yield at function calls, channel ops, etc.
- [ ] Preemptive scheduling (Go 1.14+): async preemption via signals for long-running goroutines
- [ ] Network poller: goroutines blocked on I/O are parked, not consuming an OS thread
- [ ] Scheduler tracing: `GODEBUG=schedtrace=1000`

**Garbage Collection:**
- [ ] Tri-color mark-and-sweep GC (concurrent, low-latency)
- [ ] GC phases: mark setup (STW), concurrent mark, mark termination (STW), sweep
- [ ] GC pauses: typically < 1ms in modern Go
- [ ] `GOGC`: GC target percentage (default 100 = GC when heap doubles)
- [ ] `GOMEMLIMIT` (Go 1.19+): soft memory limit, GC respects this
- [ ] `runtime.GC()`: force a GC cycle (rarely needed)
- [ ] GC tuning: `GOGC=off` for benchmarks, `GOMEMLIMIT` for containers
- [ ] Object pooling (`sync.Pool`) to reduce GC pressure
- [ ] `runtime.ReadMemStats()`: heap stats, GC stats

**Memory Model:**
- [ ] Go memory model: defines when reads in one goroutine observe writes from another
- [ ] Happens-before relationship: channel send happens before receive, mutex unlock happens before lock
- [ ] Data races: concurrent read and write to same variable without synchronization
- [ ] `-race` flag: compile-time race detector (use in tests, CI)
- [ ] `GORACE` environment variable for race detector options

**Escape Analysis:**
- [ ] Stack vs heap allocation
- [ ] Escape analysis: compiler decides whether a variable escapes to the heap
- [ ] `go build -gcflags='-m'`: see escape analysis decisions
- [ ] Variables that escape: returned pointers, interface assignments, closures capturing vars
- [ ] Reducing allocations: avoid unnecessary pointers, pre-allocate slices/maps
- [ ] Stack allocation is free (no GC needed); heap allocation has GC cost

**Performance Tuning and Profiling:**
- [ ] `pprof`: CPU profile, memory profile, goroutine profile, block profile, mutex profile
- [ ] `net/http/pprof`: HTTP endpoint for live profiling
- [ ] `go tool pprof`: analyze profiles (top, list, web, flame graph)
- [ ] `go test -bench`: benchmarking
- [ ] `go test -benchmem`: allocation counting
- [ ] `testing.B`: benchmark functions, `b.N`, `b.ResetTimer()`, `b.StopTimer()`
- [ ] `runtime/trace`: execution tracer for detailed goroutine analysis
- [ ] `go tool trace`: visualize execution traces
- [ ] Flame graphs for CPU profile visualization
- [ ] Common performance optimizations:
  - Pre-allocate slices: `make([]T, 0, expectedCap)`
  - Use `strings.Builder` for string concatenation
  - Avoid allocations in hot paths
  - Use `sync.Pool` for frequently allocated objects
  - Minimize interface conversions (boxing)
  - Use value types where possible (avoid pointer indirection)

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Profiling CLI tools, optimizing build times, understanding GC pauses in long-running services. |
| **Platform Engineering** | Performance tuning platform services, GC tuning for latency-sensitive services, pprof for production debugging. |
| **Kubernetes** | Controller performance, informer memory usage, reconciliation latency. Understanding GC is critical for controllers managing thousands of resources. |
| **MLOps** | Data pipeline throughput optimization, memory profiling for large dataset processing. |
| **LLM Infrastructure** | Inference latency optimization, memory management for token caches, GC tuning for low-latency serving. |
| **Cloud Infrastructure** | Optimizing infrastructure tools for large fleets, reducing memory footprint, profiling API gateways. |
| **GPU ML Systems** | CGO interop performance, memory management for GPU buffer allocation, reducing GC interference with GPU operations. |
| **Distributed Systems** | Tail latency optimization, GC pause impact on consensus protocols, profiling RPC handlers. |

### Practical Exercises

- [ ] Set `GOMAXPROCS(1)` and observe goroutine behavior
- [ ] Use `GODEBUG=schedtrace=1000` to see scheduler activity
- [ ] Use `runtime.ReadMemStats()` to monitor heap usage before and after operations
- [ ] Run the race detector on a concurrent program: `go run -race`
- [ ] Profile a CPU-intensive program with `pprof` and identify the hot function
- [ ] Profile memory with `pprof` and identify the top allocating function
- [ ] Use escape analysis (`-gcflags='-m'`) to see what escapes to the heap
- [ ] Write a benchmark comparing stack allocation vs heap allocation
- [ ] Tune `GOGC` and `GOMEMLIMIT` for a containerized service
- [ ] Generate a flame graph from a CPU profile

### Mini Projects

- [ ] **Performance Optimization Lab**: Take a deliberately inefficient Go program and optimize it:
  - Profile CPU, memory, goroutines
  - Identify bottlenecks (allocations, locks, GC)
  - Apply optimizations (pre-allocate, pool, reduce allocations)
  - Benchmark before and after each optimization
  - Document findings
- [ ] **Profiling HTTP Server**: Build an HTTP server with `pprof` endpoints enabled:
  - Load test with `hey` or `wrk`
  - Capture CPU profile under load
  - Capture memory profile
  - Capture goroutine profile
  - Generate flame graphs
  - Identify and fix performance issues

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Contains Duplicate | LeetCode #217 | Easy | Hash map vs sort trade-off (memory vs CPU) |
| Two Sum | LeetCode #1 | Easy | O(n) vs O(n^2) — profiling matters |
| Maximum Subarray | LeetCode #53 | Medium | Kadane's — O(n) optimal |
| Trapping Rain Water | LeetCode #42 | Hard | Multiple approaches with different memory profiles |
| Median of Two Sorted Arrays | LeetCode #4 | Hard | Algorithm efficiency |

### Interview Focused Notes

**Common Interview Questions:**
- Explain the GMP model of the Go scheduler.
- How does Go's garbage collector work? What kind is it?
- What is escape analysis? Why does it matter for performance?
- How do you profile a Go program in production?
- What is `GOGC`? How would you tune GC for a low-latency service?
- What is the Go memory model? What is a happens-before relationship?
- How does the race detector work?
- What is `GOMEMLIMIT` and when would you use it?

**Common Mistakes:**
- Premature optimization (profile first, optimize second)
- Not using the race detector in CI (missing data races)
- Setting `GOMAXPROCS` lower than number of cores (leaving performance on the table)
- Not understanding GC impact on tail latency
- Over-allocating: creating slices/maps without capacity hints
- Using pprof in production without security (expose only on internal port)

**Interviewer Expectations:**
- Can explain GMP model clearly
- Knows GC internals at a high level
- Can profile a Go application and interpret results
- Understands escape analysis and its implications
- Uses race detector consistently
- Can tune GC for different workloads

---

---

# STAGE 6: GO FOR SYSTEMS PROGRAMMING

---

## 6.1 OS Interactions and File Systems

### Concepts to Learn

- [ ] `os` package: file operations, environment, process info
- [ ] File operations: `os.Open`, `os.Create`, `os.OpenFile`, `os.Remove`, `os.Rename`
- [ ] File flags: `os.O_RDONLY`, `os.O_WRONLY`, `os.O_RDWR`, `os.O_APPEND`, `os.O_CREATE`, `os.O_TRUNC`
- [ ] File permissions: `os.FileMode`, `0644`, `0755`
- [ ] Reading files: `os.ReadFile` (whole file), `bufio.Scanner` (line by line), `io.Reader`
- [ ] Writing files: `os.WriteFile`, `bufio.Writer`, `io.Writer`
- [ ] Streaming I/O: `io.Copy`, `io.TeeReader`, `io.Pipe`, `io.MultiReader`, `io.MultiWriter`
- [ ] `io.Reader` and `io.Writer` composition: wrapping, chaining
- [ ] Temporary files: `os.CreateTemp`, `os.MkdirTemp`
- [ ] Directory operations: `os.Mkdir`, `os.MkdirAll`, `os.ReadDir`, `filepath.Walk`, `filepath.WalkDir`
- [ ] File info: `os.Stat`, `os.Lstat` (for symlinks), `fs.FileInfo`
- [ ] `path/filepath`: `Join`, `Dir`, `Base`, `Ext`, `Abs`, `Rel`, `Match`, `Glob`
- [ ] `embed` package (Go 1.16+): embed files in binary (`//go:embed`)
- [ ] `io/fs` package: filesystem abstraction
- [ ] `os.DirFS`: create `fs.FS` from directory
- [ ] `fstest.MapFS`: in-memory filesystem for testing
- [ ] File locking: `syscall.Flock` (advisory locks on Linux)
- [ ] Watching files: `fsnotify` library for file system events
- [ ] Memory-mapped files: `mmap` via syscall or library
- [ ] Large file processing: streaming instead of loading into memory

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Log file processing, config file management, build artifact handling, template rendering to files. |
| **Platform Engineering** | Configuration file management, secret file mounting, certificate management, asset embedding. |
| **Kubernetes** | ConfigMap/Secret volume mounts are files. Log file collection. Container filesystem interaction. |
| **MLOps** | Dataset file handling, model checkpoint I/O, training log parsing, artifact storage. |
| **LLM Infrastructure** | Model weight loading (large files), tokenizer vocabulary files, prompt template files. |
| **Cloud Infrastructure** | Terraform state files, provider config files, IaC template processing. |
| **GPU ML Systems** | CUDA library loading, GPU profile data, training data pipeline I/O. |
| **Distributed Systems** | WAL (write-ahead log) files, snapshot files, configuration files, data file management. |

### Practical Exercises

- [ ] Read a file line by line, count lines, words, and bytes (like `wc`)
- [ ] Implement `tail -f`: watch a file and print new lines as they're appended
- [ ] Copy a large file using `io.Copy` with a progress indicator
- [ ] Use `filepath.WalkDir` to find all Go files in a directory tree
- [ ] Embed static files using `//go:embed` and serve them via HTTP
- [ ] Use `io.Pipe` to stream data between a producer and consumer goroutine
- [ ] Process a 1GB CSV file line by line without loading it all into memory
- [ ] Implement file locking to prevent concurrent writes

### Mini Projects

- [ ] **Log Rotation System**: Build a log rotation tool that:
  - Watches a log file for growth
  - Rotates when size exceeds threshold
  - Compresses rotated files with gzip
  - Deletes files older than N days
  - Signal handling for immediate rotation
- [ ] **File Sync Tool**: Build a basic file sync (like simplified rsync):
  - Compare source and destination directories
  - Copy new/modified files (based on checksum)
  - Delete files in destination that don't exist in source (optional)
  - Progress reporting
- [ ] **Embedded Config Server**: HTTP server that serves embedded configuration files with template rendering

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Read N Characters Given Read4 | LeetCode #157 | Easy | Buffered reading |
| Read N Characters Given Read4 II | LeetCode #158 | Hard | Stateful streaming |
| Design Log Storage System | LeetCode #635 | Medium | Log file operations |
| Serialize and Deserialize Binary Tree | LeetCode #297 | Hard | File-like serialization |

### Interview Focused Notes

**Common Interview Questions:**
- How do you read a large file without loading it all into memory?
- What is `io.Reader` and why is it important?
- How do you embed files in a Go binary?
- What is `filepath.WalkDir` and how does it differ from `Walk`?
- How do you handle file locking in Go?
- What is `io.Copy` and when would you use it?
- How do you watch for file system changes?

**Common Mistakes:**
- Not closing files (resource leak) — always `defer f.Close()`
- Reading entire large files into memory
- Not handling partial reads (`io.ReadFull` vs `Read`)
- Using `filepath.Join` with user input without sanitization (path traversal vulnerability)
- Not checking errors from `Close()` on write operations (data loss)

**Interviewer Expectations:**
- Comfortable with `io.Reader`/`io.Writer` composition
- Handles large files with streaming
- Knows embed, filepath, and fs packages
- Proper resource cleanup with defer

---

## 6.2 Process Management and Signals

### Concepts to Learn

- [ ] `os/exec` package: running external commands
- [ ] `exec.Command`: creating a command
- [ ] `cmd.Run()`, `cmd.Output()`, `cmd.CombinedOutput()`, `cmd.Start()`/`cmd.Wait()`
- [ ] Stdin, Stdout, Stderr: `cmd.Stdin`, `cmd.Stdout`, `cmd.Stderr`
- [ ] Piping commands: connect stdout of one to stdin of another
- [ ] Command with context: `exec.CommandContext(ctx, ...)` for timeout/cancellation
- [ ] Environment variables: `cmd.Env`, `os.Environ()`, `os.Getenv()`, `os.Setenv()`
- [ ] Working directory: `cmd.Dir`
- [ ] Exit codes: `cmd.ProcessState.ExitCode()`
- [ ] Signal handling: `os/signal` package
- [ ] `signal.Notify(ch, os.Interrupt, syscall.SIGTERM)`: register for signals
- [ ] Graceful shutdown pattern:
  ```go
  ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
  defer stop()
  ```
- [ ] `signal.NotifyContext` (Go 1.16+): context that cancels on signal
- [ ] `os.Process`: process management, `Kill()`, `Signal()`, `Wait()`
- [ ] Process groups and foreground/background processes
- [ ] `syscall` package: low-level system calls (platform-specific)
- [ ] Daemon patterns in Go (backgrounding, PID files)

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Running external tools (git, docker, kubectl, terraform) from Go programs. Signal handling for graceful shutdown. |
| **Platform Engineering** | Process supervisors, health check executors, build pipeline runners. |
| **Kubernetes** | Container entrypoint processes must handle SIGTERM for graceful shutdown. Exec probes run commands. |
| **MLOps** | Launching training processes, running data preprocessing scripts, managing GPU processes. |
| **LLM Infrastructure** | Starting/stopping model servers, health checking inference processes. |
| **Cloud Infrastructure** | Running cloud CLI tools, managing infrastructure processes, signal-based configuration reload. |
| **GPU ML Systems** | Managing CUDA processes, monitoring GPU processes, training job lifecycle management. |
| **Distributed Systems** | Graceful shutdown is critical for data consistency. Signal handling for leader election, rebalancing. |

### Practical Exercises

- [ ] Run an external command and capture its output
- [ ] Pipe two commands together (e.g., `ls | grep .go`)
- [ ] Run a command with timeout using `exec.CommandContext`
- [ ] Implement graceful shutdown: catch SIGTERM, stop accepting work, drain existing work, exit
- [ ] Build a process supervisor that restarts a child process on crash
- [ ] Use `signal.NotifyContext` for a context that cancels on Ctrl+C
- [ ] Run a command with custom environment variables and working directory

### Mini Projects

- [ ] **Process Supervisor**: Build a process supervisor that:
  - Starts child processes from config
  - Monitors health (check PID, check port)
  - Restarts on crash with backoff
  - Forwards signals to children
  - Graceful shutdown: signals children, waits, force kills after timeout
- [ ] **Git Automation Tool**: Build a Go tool that:
  - Runs git commands (status, add, commit, push, pull)
  - Parses git output
  - Handles errors (merge conflicts, auth failures)
  - Provides a simplified API for common workflows

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Task Scheduler | LeetCode #621 | Medium | Process scheduling concepts |
| Process Tasks Using Servers | LeetCode #1882 | Medium | Priority-based scheduling |
| Single-Threaded CPU | LeetCode #1834 | Medium | Process management simulation |

### Interview Focused Notes

**Common Interview Questions:**
- How do you run external commands from Go?
- How do you handle graceful shutdown in Go?
- What is the difference between SIGTERM and SIGKILL?
- How do you implement timeout for external commands?
- What is `signal.NotifyContext`?
- How do you capture stdout AND stderr separately from a command?

**Common Mistakes:**
- Not using `CommandContext` for timeout (process hangs forever)
- Not handling SIGTERM in production services (ungraceful shutdown = data loss)
- Not collecting exit codes (can't distinguish success from failure)
- Shell injection: using `exec.Command("sh", "-c", userInput)` — use `exec.Command(binary, args...)` instead
- Not waiting for child process (zombie processes)

**Interviewer Expectations:**
- Implements graceful shutdown correctly
- Handles external command execution safely
- Understands signal propagation
- Avoids shell injection vulnerabilities

---

## 6.3 Networking: TCP, UDP, and HTTP Servers

### Concepts to Learn

**TCP:**
- [ ] `net.Listen("tcp", ":8080")`: create a TCP listener
- [ ] `listener.Accept()`: accept incoming connections (blocking)
- [ ] `net.Conn` interface: `Read`, `Write`, `Close`, `SetDeadline`
- [ ] Connection handling in goroutines (one goroutine per connection)
- [ ] Connection timeouts: `SetReadDeadline`, `SetWriteDeadline`
- [ ] TCP keep-alive: `net.TCPConn.SetKeepAlive`
- [ ] Building a simple TCP echo server
- [ ] Building a TCP chat server
- [ ] Connection pooling concepts
- [ ] Buffered I/O over TCP: `bufio.Reader`, `bufio.Writer`

**UDP:**
- [ ] `net.ListenPacket("udp", ":8080")`: create UDP listener
- [ ] `net.PacketConn`: `ReadFrom`, `WriteTo`
- [ ] UDP vs TCP: connectionless, no guaranteed delivery, lower latency
- [ ] Use cases: DNS, metrics, real-time gaming, service discovery

**HTTP Server:**
- [ ] `net/http` package: built-in HTTP server
- [ ] `http.HandleFunc`: register a handler function
- [ ] `http.Handler` interface: `ServeHTTP(w http.ResponseWriter, r *http.Request)`
- [ ] `http.ServeMux`: built-in request multiplexer (router)
- [ ] Enhanced `http.ServeMux` (Go 1.22+): method-based routing, path parameters
- [ ] `http.Server` struct: timeouts, TLS, graceful shutdown
- [ ] Server timeouts: `ReadTimeout`, `WriteTimeout`, `IdleTimeout`, `ReadHeaderTimeout`
- [ ] `http.ResponseWriter`: `Write`, `WriteHeader`, `Header`
- [ ] `http.Request`: `Method`, `URL`, `Header`, `Body`, `Context()`
- [ ] Middleware pattern: `func(http.Handler) http.Handler`
- [ ] `http.Client`: making HTTP requests, timeout, transport
- [ ] `http.Transport`: connection pooling, TLS, proxy, keep-alive
- [ ] `httptest` package: `NewServer`, `NewRequest`, `NewRecorder`
- [ ] TLS: `http.ListenAndServeTLS`, `tls.Config`
- [ ] HTTP/2: automatic with TLS in Go, `golang.org/x/net/http2`
- [ ] Graceful shutdown: `server.Shutdown(ctx)` (Go 1.8+)
- [ ] WebSocket: `gorilla/websocket` or `nhooyr.io/websocket`

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Building health check servers, webhook receivers, API servers for tooling. |
| **Platform Engineering** | Platform API servers, internal service communication, load balancer health endpoints. |
| **Kubernetes** | Admission webhooks are HTTP servers. Metrics endpoints are HTTP. API server is HTTP. |
| **MLOps** | Model serving HTTP endpoints, training dashboard servers, artifact download servers. |
| **LLM Infrastructure** | Inference HTTP/gRPC endpoints, streaming SSE endpoints, health/readiness probes. |
| **Cloud Infrastructure** | API gateways, reverse proxies, webhook handlers, TLS termination. |
| **GPU ML Systems** | GPU monitoring HTTP dashboards, device plugin gRPC servers. |
| **Distributed Systems** | Custom protocol servers over TCP, gossip protocol over UDP, RPC over HTTP. |

### Practical Exercises

- [ ] Build a TCP echo server that handles multiple clients concurrently
- [ ] Build a UDP server that receives and echoes datagrams
- [ ] Build an HTTP server with multiple routes using `http.ServeMux`
- [ ] Add middleware for logging, CORS, and request ID
- [ ] Implement graceful shutdown for an HTTP server
- [ ] Build an HTTP client with timeout, retry, and connection pooling
- [ ] Write integration tests using `httptest.NewServer`
- [ ] Build a simple reverse proxy using `httputil.NewSingleHostReverseProxy`
- [ ] Add TLS to an HTTP server with self-signed certificates
- [ ] Use Go 1.22+ enhanced mux with method routing: `mux.HandleFunc("GET /api/users/{id}", handler)`

### Mini Projects

- [ ] **TCP Chat Server**: Multi-client chat server with:
  - Room management (join, leave, list)
  - Private messages
  - Graceful disconnect handling
  - Admin commands
- [ ] **HTTP API Gateway**: Build a reverse proxy / API gateway with:
  - Route-based backend selection
  - Request/response logging middleware
  - Rate limiting middleware
  - Authentication middleware
  - Health check endpoints for backends
  - Graceful shutdown
- [ ] **WebSocket Notification Server**: Build a WebSocket server that:
  - Accepts client connections
  - Supports topic subscription
  - Broadcasts messages to subscribers
  - Handles reconnection

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Design Browser History | LeetCode #1472 | Medium | Request/response pattern |
| Design Underground System | LeetCode #1396 | Medium | Stateful server |
| Number of Recent Calls | LeetCode #933 | Easy | Rate limiting concept |
| Logger Rate Limiter | LeetCode #359 | Easy | Rate limiting |
| Design Hit Counter | LeetCode #362 | Medium | Server metrics |

### Interview Focused Notes

**Common Interview Questions:**
- How do you build an HTTP server in Go?
- What timeouts should you set on an HTTP server? Why?
- How does graceful shutdown work?
- What is the difference between `http.HandleFunc` and `http.Handler`?
- How do you implement middleware in Go?
- How does `httptest` work for testing?
- What is the difference between TCP and UDP? When would you use each?
- How do you handle concurrent connections in a TCP server?

**Common Mistakes:**
- Not setting server timeouts (Slowloris attack vulnerability)
- Not calling `resp.Body.Close()` in HTTP client (connection leak)
- Not using `context` for HTTP requests (no timeout/cancellation)
- Reading the entire request body into memory (OOM for large uploads)
- Not implementing graceful shutdown (dropped connections on deploy)

**Interviewer Expectations:**
- Can build a production-ready HTTP server from scratch
- Sets all necessary timeouts
- Implements graceful shutdown
- Writes middleware correctly
- Tests with httptest
- Understands TCP vs UDP trade-offs

---

---

# STAGE 7: GO NETWORKING DEEP DIVE

---

## 7.1 HTTP Internals and Advanced Client

### Concepts to Learn

- [ ] HTTP/1.1 protocol: request/response, headers, body, keep-alive, chunked encoding
- [ ] HTTP/2: multiplexing, server push, header compression (HPACK), binary framing
- [ ] HTTP/3 / QUIC: UDP-based, 0-RTT, built-in encryption (emerging in Go)
- [ ] `http.Transport` internals:
  - Connection pooling: `MaxIdleConns`, `MaxIdleConnsPerHost`, `MaxConnsPerHost`
  - TLS configuration: `TLSClientConfig`
  - Proxy support: `http.ProxyFromEnvironment`
  - Dial function: custom connection establishment
  - `DisableKeepAlives`, `DisableCompression`
  - `IdleConnTimeout`, `TLSHandshakeTimeout`, `ResponseHeaderTimeout`
- [ ] Custom `http.RoundTripper`: middleware for HTTP client
- [ ] Client-side middleware pattern: logging, auth, retry, tracing via RoundTripper
- [ ] Cookie handling: `http.CookieJar`
- [ ] Redirect policy: `Client.CheckRedirect`
- [ ] Request body: `io.Reader` based, streaming
- [ ] Response body: always close, streaming reads
- [ ] `io.ReadAll` vs streaming (memory trade-offs)
- [ ] Context on requests: `req.WithContext(ctx)`
- [ ] Multipart uploads: `mime/multipart`
- [ ] Server-Sent Events (SSE): streaming responses
- [ ] Content negotiation: Accept headers

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | API clients for GitHub, Slack, PagerDuty. Understanding connection pooling for high-throughput CLI tools. |
| **Platform Engineering** | Internal service clients, API gateway implementation, health check clients. |
| **Kubernetes** | `client-go` uses `http.Transport` heavily. Understanding connection behavior matters for API server communication. |
| **MLOps** | Model registry HTTP clients, artifact download clients, metrics reporting clients. |
| **LLM Infrastructure** | Streaming inference responses (SSE), model API clients, embedding service clients. |
| **Cloud Infrastructure** | Cloud provider API clients, credential refresh, retry with backoff. |
| **GPU ML Systems** | NVIDIA management API clients, GPU monitoring endpoints. |
| **Distributed Systems** | Service-to-service HTTP communication, connection pool tuning, retry logic. |

### Practical Exercises

- [ ] Create a custom `http.Transport` with connection pool tuning
- [ ] Implement a `RoundTripper` that adds authentication headers to every request
- [ ] Implement a `RoundTripper` that logs request/response timing
- [ ] Implement a `RoundTripper` that retries on 5xx errors with exponential backoff
- [ ] Build an SSE client that reads streaming events
- [ ] Build an SSE server that pushes events to connected clients
- [ ] Upload a file using multipart form data
- [ ] Stream a large download to disk without loading into memory

### Mini Projects

- [ ] **Production HTTP Client Library**: Build a reusable HTTP client with:
  - Connection pooling configuration
  - Automatic retry with exponential backoff and jitter
  - Circuit breaker integration
  - Request/response logging (with body truncation)
  - Metrics (latency histogram, error count)
  - OAuth2 token refresh
  - Context propagation
- [ ] **Server-Sent Events Platform**: Build a notification system using SSE:
  - HTTP endpoint that accepts SSE connections
  - Topic-based subscription
  - Message broadcasting
  - Client reconnection handling
  - Connection management (heartbeat, cleanup)

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Design Rate Limiter | System Design | Medium | HTTP rate limiting |
| Implement LRU Cache | LeetCode #146 | Medium | Connection pool concept |
| Design URL Shortener | System Design | Medium | HTTP service design |

### Interview Focused Notes

**Common Interview Questions:**
- How does HTTP/2 differ from HTTP/1.1?
- What is connection pooling and how does Go handle it?
- How do you implement client-side middleware in Go?
- What timeouts should you set on an HTTP client?
- How do you handle streaming responses?
- What is a RoundTripper?

**Common Mistakes:**
- Not configuring `MaxIdleConnsPerHost` (default is 2 — too low for high-throughput)
- Not closing response bodies (connection leak)
- Not setting client timeout (hangs forever on slow servers)
- Creating a new `http.Client` per request (no connection reuse)

**Interviewer Expectations:**
- Understands HTTP protocol details
- Can configure HTTP client for production use
- Implements client middleware via RoundTripper
- Handles streaming and large responses correctly

---

## 7.2 gRPC and Protocol Buffers

### Concepts to Learn

- [ ] What is gRPC: high-performance RPC framework by Google
- [ ] Protocol Buffers (protobuf): binary serialization format
- [ ] `.proto` file syntax: messages, services, enums, oneof, repeated, map
- [ ] Protobuf scalar types: `int32`, `int64`, `string`, `bytes`, `bool`, `float`, `double`
- [ ] Protobuf code generation: `protoc`, `protoc-gen-go`, `protoc-gen-go-grpc`
- [ ] `buf` tool: modern protobuf management (lint, generate, break detection)
- [ ] gRPC service types:
  - Unary: single request, single response
  - Server streaming: single request, stream of responses
  - Client streaming: stream of requests, single response
  - Bidirectional streaming: stream of requests and responses
- [ ] gRPC Go implementation: `google.golang.org/grpc`
- [ ] Server implementation: register service, listen, serve
- [ ] Client implementation: dial, create stub, call methods
- [ ] gRPC interceptors: unary and stream interceptors (middleware)
- [ ] gRPC metadata: headers, like HTTP headers
- [ ] gRPC status codes: `codes.OK`, `codes.NotFound`, `codes.Internal`, etc.
- [ ] Error handling: `status.Error`, `status.Errorf`
- [ ] gRPC deadlines and cancellation: context propagation
- [ ] gRPC health checking: `grpc.health.v1` protocol
- [ ] gRPC reflection: for tools like `grpcurl`
- [ ] gRPC load balancing: client-side vs proxy-based
- [ ] gRPC-Gateway: REST to gRPC transcoding
- [ ] `Connect` protocol (by Buf): simpler alternative to gRPC
- [ ] Performance: binary serialization, HTTP/2, multiplexing

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Many internal tools communicate via gRPC. Understanding protobuf for config and state. |
| **Platform Engineering** | Internal service mesh communication, API contracts, schema evolution. |
| **Kubernetes** | Kubernetes uses gRPC internally (kubelet to CRI, CSI, device plugins). CRI is a gRPC API. |
| **MLOps** | Model serving often uses gRPC (Triton, TensorFlow Serving). Pipeline step communication. |
| **LLM Infrastructure** | High-performance inference APIs use gRPC for low latency. Streaming token generation uses server streaming. |
| **Cloud Infrastructure** | Cloud provider APIs increasingly use gRPC. Terraform plugins use gRPC. |
| **GPU ML Systems** | NVIDIA device plugin uses gRPC. GPU scheduler communication. |
| **Distributed Systems** | gRPC is the standard for inter-service communication. Consensus protocols often use gRPC. |

### Practical Exercises

- [ ] Install `protoc`, `protoc-gen-go`, `protoc-gen-go-grpc` (or use `buf`)
- [ ] Define a `.proto` file for a user service with CRUD operations
- [ ] Generate Go code from the proto file
- [ ] Implement the gRPC server
- [ ] Implement the gRPC client
- [ ] Add a server streaming RPC (e.g., list users with pagination)
- [ ] Add unary interceptors for logging and authentication
- [ ] Add stream interceptors for logging
- [ ] Use `grpcurl` to test the server
- [ ] Implement gRPC health checking
- [ ] Add deadline/timeout to client calls

### Mini Projects

- [ ] **gRPC Microservice**: Build a complete gRPC service with:
  - Proto definitions with `buf`
  - Unary and streaming RPCs
  - Interceptors: auth, logging, metrics, recovery
  - Health checking
  - Reflection for debugging
  - Integration tests
  - gRPC-Gateway for REST compatibility
- [ ] **gRPC Chat Service**: Build a chat service using bidirectional streaming:
  - User authentication
  - Room management
  - Message broadcasting
  - Online user tracking
  - Message history

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Serialize and Deserialize Binary Tree | LeetCode #297 | Hard | Serialization concepts |
| Encode and Decode TinyURL | LeetCode #535 | Medium | Service design |
| Design Phone Directory | LeetCode #379 | Medium | Service interface design |

### Interview Focused Notes

**Common Interview Questions:**
- What is gRPC and how does it differ from REST?
- What are the four types of gRPC service methods?
- What is protobuf and why is it used with gRPC?
- How do you handle errors in gRPC?
- What are gRPC interceptors?
- How does gRPC handle deadlines and cancellation?
- When would you choose gRPC over REST?
- What is gRPC health checking?

**Common Mistakes:**
- Not handling context cancellation in gRPC servers
- Not setting deadlines on client calls (hangs forever)
- Breaking protobuf backward compatibility (changing field numbers)
- Not implementing health checking (load balancer can't route traffic)
- Large messages without streaming (protobuf has message size limits)

**Interviewer Expectations:**
- Can design and implement a gRPC service end-to-end
- Understands protobuf schema evolution rules
- Implements interceptors for cross-cutting concerns
- Knows when to use streaming vs unary
- Handles errors with proper gRPC status codes

---

## 7.3 Service Communication Patterns

### Concepts to Learn

- [ ] Synchronous communication: REST, gRPC (request-response)
- [ ] Asynchronous communication: message queues, event streaming
- [ ] Service discovery:
  - Client-side: service registry (Consul, etcd), client queries registry
  - Server-side: load balancer (Kubernetes Service), client connects to LB
  - DNS-based: Kubernetes CoreDNS, `<service>.<namespace>.svc.cluster.local`
- [ ] Load balancing:
  - Client-side: gRPC built-in, custom resolvers
  - Server-side: Kubernetes Service, Envoy, nginx
  - Algorithms: round-robin, least connections, random, weighted, consistent hashing
- [ ] Circuit breaker pattern: prevent cascading failures
- [ ] Retry patterns: exponential backoff with jitter, retry budget
- [ ] Timeout patterns: per-call, per-attempt, overall deadline
- [ ] Bulkhead pattern: isolate failures to prevent system-wide impact
- [ ] Sidecar pattern: service mesh (Istio, Linkerd)
- [ ] API gateway pattern: single entry point for external traffic
- [ ] Backend-for-Frontend (BFF) pattern
- [ ] Saga pattern for distributed transactions
- [ ] Outbox pattern for reliable event publishing
- [ ] Idempotency: making operations safe to retry
- [ ] Request deduplication

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Setting up service communication, configuring load balancers, managing API gateways. |
| **Platform Engineering** | Designing the communication layer, choosing patterns, implementing service mesh. |
| **Kubernetes** | Kubernetes provides Service discovery and load balancing. Understanding these patterns is core. |
| **MLOps** | Pipeline steps communicate via APIs. Model serving behind load balancers. |
| **LLM Infrastructure** | Inference load balancing, model routing, request queuing, failover. |
| **Cloud Infrastructure** | Multi-service architectures, cross-region communication, API management. |
| **GPU ML Systems** | GPU-aware load balancing, model replica management. |
| **Distributed Systems** | Communication patterns ARE distributed systems. Every pattern listed is fundamental. |

### Practical Exercises

- [ ] Implement a service registry client that registers/deregisters a service
- [ ] Build a client-side load balancer with round-robin and least-connections strategies
- [ ] Implement a circuit breaker with open/half-open/closed states
- [ ] Build an idempotent API endpoint using idempotency keys
- [ ] Implement retry with exponential backoff and jitter
- [ ] Build a simple API gateway that routes to multiple backends

### Mini Projects

- [ ] **Service Mesh Lite**: Build a lightweight service mesh sidecar with:
  - Service registration with Consul or etcd
  - Client-side load balancing
  - Circuit breaker per backend
  - Retry with backoff
  - Request logging and metrics
  - Health checking
- [ ] **Saga Orchestrator**: Build a saga orchestrator for a multi-step workflow:
  - Step execution with compensating actions
  - Distributed transaction coordination
  - State persistence
  - Failure recovery

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Design Consistent Hashing | System Design | Hard | Load balancing fundamental |
| Implement Queue with Limited Size | Custom | Medium | Bulkhead pattern |
| Random Pick with Weight | LeetCode #528 | Medium | Weighted load balancing |
| Design Leaderboard | LeetCode #1244 | Medium | Service registry concept |

### Interview Focused Notes

**Common Interview Questions:**
- What service communication patterns do you know?
- When would you use synchronous vs asynchronous communication?
- How does service discovery work in Kubernetes?
- What is a circuit breaker and when would you use it?
- How do you implement retry safely? What is idempotency?
- What is the saga pattern?
- How does a service mesh work?

**Common Mistakes:**
- Retry without idempotency (duplicate actions)
- Retry without backoff (thundering herd)
- No circuit breaker (cascading failures)
- Synchronous communication for everything (tight coupling, latency chain)
- Not setting timeouts (one slow service blocks everything)

**Interviewer Expectations:**
- Knows when to use sync vs async communication
- Can implement resilience patterns (circuit breaker, retry, timeout)
- Understands service discovery mechanisms
- Can design for failure in distributed communication

---

---

---

# STAGE 8: API DEVELOPMENT (Production-Grade)

---

## 8.1 REST API Design Principles

### Concepts to Learn

- [ ] REST architectural constraints: client-server, stateless, cacheable, uniform interface, layered, code-on-demand
- [ ] Resource naming: nouns not verbs, plural (`/users`, `/users/{id}`)
- [ ] HTTP methods: GET (read), POST (create), PUT (full update), PATCH (partial update), DELETE
- [ ] HTTP status codes: 200, 201, 204, 400, 401, 403, 404, 409, 422, 429, 500, 502, 503
- [ ] Request/response formats: JSON, content type headers
- [ ] Pagination: offset-based, cursor-based (cursor is preferred for large datasets)
- [ ] Filtering, sorting, and field selection via query parameters
- [ ] HATEOAS (Hypermedia as the Engine of Application State) — know the concept
- [ ] API versioning strategies: URL path (`/v1/`), header, query parameter
- [ ] Idempotency keys for POST requests
- [ ] ETags and conditional requests (If-None-Match, If-Match)
- [ ] Bulk operations: batch create, batch update, batch delete
- [ ] Partial responses: field selection (`?fields=id,name,email`)
- [ ] Error response format: consistent structure `{ "error": { "code": "...", "message": "..." } }`
- [ ] OpenAPI / Swagger specification
- [ ] API documentation generation

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Internal APIs for automation, webhook endpoints, tool APIs. |
| **Platform Engineering** | Platform APIs are the product. API design quality directly impacts user experience. |
| **Kubernetes** | Kubernetes API follows REST conventions. Understanding API design helps when extending the API. |
| **MLOps** | Model serving APIs, experiment tracking APIs, pipeline trigger APIs. |
| **LLM Infrastructure** | Inference APIs (OpenAI-compatible), prompt management APIs, model management APIs. |
| **Cloud Infrastructure** | Cloud provider APIs follow REST. Building Terraform providers requires understanding API design. |
| **GPU ML Systems** | GPU management APIs, device allocation APIs, training job APIs. |
| **Distributed Systems** | Admin APIs, health APIs, cluster management APIs. |

### Practical Exercises

- [ ] Design a REST API for a project management system (resources: projects, tasks, users, comments)
- [ ] Implement cursor-based pagination
- [ ] Implement filtering and sorting via query parameters
- [ ] Create an OpenAPI spec for your API
- [ ] Implement ETags for conditional requests
- [ ] Design consistent error responses across all endpoints
- [ ] Add API versioning via URL path prefix

### Mini Projects

- [ ] **RESTful CRUD API**: Build a complete REST API with:
  - CRUD for multiple resources
  - Proper HTTP methods and status codes
  - Cursor-based pagination
  - Filtering and sorting
  - Consistent error responses
  - Request validation
  - OpenAPI documentation
  - Integration tests

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Design HashMap | LeetCode #706 | Easy | Key-value store concept |
| Time Based Key-Value Store | LeetCode #981 | Medium | Versioned API concept |
| Design a Stack With Increment Operation | LeetCode #1381 | Medium | API operation design |

### Interview Focused Notes

**Common Interview Questions:**
- What makes a good REST API?
- When would you use PUT vs PATCH?
- How do you handle pagination in a REST API?
- What is the difference between cursor-based and offset-based pagination?
- How do you version APIs?
- How do you design error responses?

**Common Mistakes:**
- Using verbs in URLs (`/getUser`) instead of nouns (`/users/{id}`)
- Returning 200 for everything (including errors)
- Not implementing pagination (returning all records)
- Inconsistent error response format across endpoints
- Not using proper HTTP methods (using POST for everything)

**Interviewer Expectations:**
- Designs clean, consistent REST APIs
- Uses correct HTTP methods and status codes
- Implements proper pagination, filtering, and error handling
- Can explain API versioning strategies

---

## 8.2 Go Web Frameworks

### Concepts to Learn

**Standard Library (`net/http`):**
- [ ] Enhanced ServeMux (Go 1.22+): method-based routing, path parameters
- [ ] Advantages: zero dependencies, full control, matches Go philosophy
- [ ] When to use: simple services, internal APIs, maximum control

**Gin:**
- [ ] Router: path parameters, query parameters, grouped routes
- [ ] Middleware: built-in (Logger, Recovery) and custom
- [ ] Request binding: JSON, XML, form, query, header validation
- [ ] Response rendering: JSON, XML, YAML, HTML
- [ ] Validation: `binding:"required,email"`
- [ ] Error handling: `c.Error()`, custom error handlers
- [ ] File upload handling
- [ ] Performance: fastest router (radix tree based)

**Echo:**
- [ ] Router, middleware, request binding, response rendering
- [ ] Built-in middleware: CORS, JWT, rate limiting, request ID
- [ ] Custom error handler
- [ ] Data binding and validation
- [ ] WebSocket support

**Fiber:**
- [ ] Built on `fasthttp` (not `net/http`)
- [ ] Express.js-like API
- [ ] High performance
- [ ] Limitation: not compatible with `net/http` middleware

**Chi:**
- [ ] Compatible with `net/http` (uses standard interfaces)
- [ ] Middleware stack: composable, reusable
- [ ] URL parameters, sub-routers
- [ ] Lightweight, zero external dependencies
- [ ] Best for composing `net/http` compatible middleware

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Internal API servers for automation tools, webhook receivers. |
| **Platform Engineering** | Platform APIs serving thousands of requests. Framework choice impacts team productivity. |
| **Kubernetes** | Webhook servers (admission, conversion) often use Gin or net/http. Health endpoints. |
| **MLOps** | Model serving APIs, experiment tracking servers, pipeline management APIs. |
| **LLM Infrastructure** | High-throughput inference APIs, often using Gin or Fiber for performance. |
| **Cloud Infrastructure** | Cloud management APIs, provider interfaces. |
| **GPU ML Systems** | GPU management APIs, monitoring dashboards. |
| **Distributed Systems** | Admin APIs, health check endpoints, management consoles. |

### Practical Exercises

- [ ] Build the same CRUD API in: (a) net/http, (b) Gin, (c) Chi — compare code
- [ ] Implement custom middleware in Gin: request logging, auth, rate limiting
- [ ] Implement request validation with Gin's binding tags
- [ ] Build a file upload endpoint with progress tracking
- [ ] Benchmark all four frameworks under load with `hey` or `wrk`
- [ ] Implement sub-routers for API versioning in Chi

### Mini Projects

- [ ] **Full-Featured REST API**: Using Gin or Chi, build:
  - User management (register, login, profile, CRUD)
  - JWT authentication
  - Role-based authorization
  - Request validation
  - Cursor-based pagination
  - Rate limiting
  - Request logging with structured logs
  - Health and readiness endpoints
  - Graceful shutdown
  - Comprehensive integration tests
  - Docker container

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Design URL Shortener | System Design | Medium | Full API design exercise |
| Design Pastebin | System Design | Medium | API with file handling |

### Interview Focused Notes

**Common Interview Questions:**
- Which Go web framework do you prefer and why?
- When would you use `net/http` vs a framework?
- How do you implement middleware in Go?
- How do you handle request validation?
- How do you structure a Go web application?
- What is the difference between Gin and Chi in terms of compatibility?

**Common Mistakes:**
- Choosing a framework without understanding the tradeoffs
- Not implementing middleware for cross-cutting concerns
- Not validating request input (security vulnerability)
- Not implementing graceful shutdown
- Putting business logic in HTTP handlers (should be in service layer)

**Interviewer Expectations:**
- Can build a production API with any framework
- Separates handler logic from business logic
- Implements proper middleware chain
- Validates all input
- Tests API endpoints thoroughly

---

## 8.3 Authentication, Authorization, and API Security

### Concepts to Learn

- [ ] Authentication vs Authorization
- [ ] Session-based auth: cookies, server-side sessions
- [ ] Token-based auth: JWT (JSON Web Tokens)
- [ ] JWT structure: header, payload, signature
- [ ] JWT claims: `iss`, `sub`, `exp`, `iat`, `aud`, custom claims
- [ ] JWT signing algorithms: HS256 (symmetric), RS256 (asymmetric)
- [ ] JWT refresh tokens: short-lived access tokens, long-lived refresh tokens
- [ ] OAuth 2.0 flows:
  - Authorization Code (web apps)
  - Authorization Code + PKCE (SPAs, mobile)
  - Client Credentials (service-to-service)
  - Refresh Token
- [ ] OpenID Connect (OIDC): identity layer on top of OAuth 2.0
- [ ] API keys: simple but less secure
- [ ] mTLS: mutual TLS for service-to-service auth
- [ ] RBAC: Role-Based Access Control
- [ ] ABAC: Attribute-Based Access Control
- [ ] Middleware for authentication and authorization
- [ ] Rate limiting: per-user, per-IP, per-API key
  - Token bucket algorithm
  - Sliding window algorithm
  - Fixed window algorithm
- [ ] CORS: Cross-Origin Resource Sharing
- [ ] CSRF protection
- [ ] Input validation and sanitization
- [ ] SQL injection prevention (parameterized queries)
- [ ] XSS prevention (output encoding)
- [ ] Security headers: Content-Security-Policy, X-Frame-Options, HSTS
- [ ] API abuse prevention: rate limiting, request size limits, timeout

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Securing automation APIs, managing secrets, implementing service-to-service auth. |
| **Platform Engineering** | Platform APIs need robust auth. Multi-tenant authorization. API key management. |
| **Kubernetes** | RBAC for API access, service accounts, admission webhook authentication, mTLS in service mesh. |
| **MLOps** | Securing model serving endpoints, experiment tracking auth, pipeline authorization. |
| **LLM Infrastructure** | API key management for inference endpoints, rate limiting per customer, usage tracking. |
| **Cloud Infrastructure** | IAM, service accounts, credential rotation, API security. |
| **GPU ML Systems** | Access control for GPU resources, authorization for training job submission. |
| **Distributed Systems** | Service-to-service authentication, authorization in distributed systems, zero-trust networking. |

### Practical Exercises

- [ ] Implement JWT authentication: login endpoint returns tokens, middleware validates tokens
- [ ] Implement refresh token rotation
- [ ] Build RBAC middleware: define roles and permissions, check on each request
- [ ] Implement rate limiting middleware (token bucket algorithm)
- [ ] Implement API key authentication
- [ ] Set up CORS middleware with proper configuration
- [ ] Implement OAuth 2.0 client credentials flow for service-to-service auth
- [ ] Add security headers middleware

### Mini Projects

- [ ] **Auth Service**: Build a complete authentication/authorization service:
  - User registration with email verification
  - Login with JWT (access + refresh tokens)
  - Token refresh endpoint
  - Password reset flow
  - RBAC with roles and permissions
  - API key management (create, revoke, list)
  - Rate limiting per user/API key
  - Audit logging
- [ ] **API Gateway with Auth**: Build an API gateway that:
  - Validates JWT tokens
  - Checks RBAC permissions per route
  - Rate limits per API key
  - Adds security headers
  - Logs all access attempts
  - Proxies to backend services

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Logger Rate Limiter | LeetCode #359 | Easy | Rate limiting |
| Design Authentication Manager | LeetCode #1797 | Medium | Token expiration |
| Number of Recent Calls | LeetCode #933 | Easy | Sliding window for rate limiting |

### Interview Focused Notes

**Common Interview Questions:**
- How does JWT authentication work?
- What is the difference between authentication and authorization?
- How do you implement rate limiting?
- What is OAuth 2.0? Explain the client credentials flow.
- How do you secure service-to-service communication?
- What is RBAC and how do you implement it?
- How do you prevent common web vulnerabilities (XSS, CSRF, SQL injection)?

**Common Mistakes:**
- Storing sensitive data in JWT payload (it's base64 encoded, not encrypted)
- Not implementing token expiration
- Not using HTTPS (tokens transmitted in plain text)
- Rate limiting only by IP (doesn't work behind NAT/proxy)
- Hardcoding secrets in code
- Not validating JWT signature algorithm (algorithm confusion attack)

**Interviewer Expectations:**
- Can implement JWT auth end-to-end
- Understands OAuth 2.0 flows
- Implements rate limiting correctly
- Knows common security vulnerabilities and prevention
- Designs authorization at the middleware level

---

---

# STAGE 9: DATABASES AND DATA PERSISTENCE

---

## 9.1 SQL Fundamentals and PostgreSQL

### Concepts to Learn

- [ ] SQL basics: SELECT, INSERT, UPDATE, DELETE, WHERE, ORDER BY, GROUP BY, HAVING
- [ ] Joins: INNER, LEFT, RIGHT, FULL OUTER, CROSS, self-join
- [ ] Subqueries: scalar, correlated, EXISTS, IN
- [ ] Aggregations: COUNT, SUM, AVG, MIN, MAX
- [ ] Window functions: ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD, OVER, PARTITION BY
- [ ] Common Table Expressions (CTEs): WITH clause, recursive CTEs
- [ ] Indexing: B-tree, hash, GIN, GiST, partial indexes, covering indexes
- [ ] EXPLAIN ANALYZE: query plan analysis
- [ ] Transactions: ACID properties, isolation levels (READ COMMITTED, REPEATABLE READ, SERIALIZABLE)
- [ ] Deadlocks: detection, prevention, ordering
- [ ] Connection pooling: why it matters, pgBouncer
- [ ] Schema design: normalization (1NF, 2NF, 3NF), denormalization trade-offs
- [ ] Migrations: schema versioning, migration tools
- [ ] PostgreSQL-specific: JSONB, arrays, enums, generated columns, partitioning
- [ ] PostgreSQL extensions: `uuid-ossp`, `pg_trgm`, `postgis`
- [ ] Database performance: query optimization, index selection, vacuum, analyze

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Managing databases, running migrations, monitoring query performance. |
| **Platform Engineering** | Platform metadata storage, tenant management, resource catalog, audit logs. |
| **Kubernetes** | etcd stores Kubernetes state, but platform services use PostgreSQL. Operators often manage databases. |
| **MLOps** | Experiment tracking, model metadata, pipeline state, feature store. |
| **LLM Infrastructure** | Prompt storage, conversation history, usage tracking, model registry. |
| **Cloud Infrastructure** | Resource inventory, state management, configuration storage. |
| **GPU ML Systems** | GPU allocation records, training job history, resource usage tracking. |
| **Distributed Systems** | State storage, distributed locks, leader election with advisory locks. |

### Practical Exercises

- [ ] Design a schema for a task management system (users, projects, tasks, comments, labels)
- [ ] Write complex queries with joins, window functions, and CTEs
- [ ] Use EXPLAIN ANALYZE to optimize a slow query
- [ ] Create appropriate indexes and measure performance improvement
- [ ] Implement a database migration using `golang-migrate`
- [ ] Write a transaction that transfers money between accounts (demonstrate ACID)
- [ ] Set up PostgreSQL with connection pooling

### Mini Projects

- [ ] **Database Performance Lab**: Create a table with 1M rows, write various queries, use EXPLAIN ANALYZE to optimize them, add indexes, measure improvement

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Combine Two Tables | LeetCode SQL #175 | Easy | JOIN |
| Second Highest Salary | LeetCode SQL #176 | Medium | Subquery |
| Rank Scores | LeetCode SQL #178 | Medium | Window functions |
| Department Top Three Salaries | LeetCode SQL #185 | Hard | Window functions + CTE |
| Consecutive Numbers | LeetCode SQL #180 | Medium | Self-join / window |

### Interview Focused Notes

**Common Interview Questions:**
- Explain ACID properties
- What are transaction isolation levels? When would you use SERIALIZABLE?
- How do indexes work? When would you NOT add an index?
- How do you detect and resolve deadlocks?
- What is the N+1 query problem?
- How do you optimize a slow query?

**Common Mistakes:**
- N+1 query problem (loading related data in a loop)
- Missing indexes on foreign keys
- Not using parameterized queries (SQL injection)
- Over-indexing (slows writes)
- Not using transactions for multi-step operations

**Interviewer Expectations:**
- Writes correct, optimized SQL
- Understands indexing strategies
- Can analyze query plans
- Knows transaction isolation levels
- Designs normalized schemas with appropriate denormalization

---

## 9.2 Go Database Libraries: GORM, sqlx, pgx

### Concepts to Learn

**`database/sql` (Standard Library):**
- [ ] `sql.Open`, `sql.DB` (connection pool, not a single connection)
- [ ] `db.Query`, `db.QueryRow`, `db.Exec`
- [ ] `rows.Scan` for reading results
- [ ] Prepared statements: `db.Prepare`
- [ ] Transactions: `db.Begin`, `tx.Commit`, `tx.Rollback`
- [ ] Context-aware methods: `QueryContext`, `ExecContext`
- [ ] Connection pool tuning: `SetMaxOpenConns`, `SetMaxIdleConns`, `SetConnMaxLifetime`, `SetConnMaxIdleTime`
- [ ] `sql.NullString`, `sql.NullInt64` for nullable columns

**pgx (PostgreSQL driver):**
- [ ] `pgx` vs `lib/pq`: pgx is newer, faster, more features
- [ ] `pgxpool`: connection pooling
- [ ] Native PostgreSQL types: JSONB, arrays, enums, UUID
- [ ] Batch queries: `pgx.Batch`
- [ ] COPY protocol for bulk inserts
- [ ] `pgx.ConnConfig` for connection configuration
- [ ] Listen/Notify for PostgreSQL pub-sub

**sqlx (Extension of database/sql):**
- [ ] `sqlx.DB` wraps `sql.DB`
- [ ] `StructScan`: scan rows directly into structs
- [ ] `NamedExec`, `NamedQuery`: named parameters
- [ ] `Get` (single row) and `Select` (multiple rows)
- [ ] `In` clause expansion: `sqlx.In`
- [ ] `sqlx.Named`: named query preparation

**GORM (ORM):**
- [ ] Model definition with struct tags
- [ ] Auto-migration: `db.AutoMigrate(&User{})`
- [ ] CRUD operations: `Create`, `First`, `Find`, `Save`, `Delete`
- [ ] Associations: `HasOne`, `HasMany`, `BelongsTo`, `ManyToMany`
- [ ] Hooks: `BeforeCreate`, `AfterCreate`, etc.
- [ ] Scopes: reusable query conditions
- [ ] Preloading: eager loading associations
- [ ] Raw SQL escape hatch
- [ ] GORM Gen: type-safe query builder (code generation)
- [ ] Transactions in GORM
- [ ] When to use GORM vs raw SQL: GORM for CRUD-heavy apps, raw SQL for complex queries

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Database automation, migration tooling, backup/restore utilities. |
| **Platform Engineering** | Platform data layer, multi-tenant data access, data migration tools. |
| **Kubernetes** | Operators that manage databases, custom controllers that persist state to PostgreSQL. |
| **MLOps** | Experiment tracking storage, model metadata persistence, feature store backend. |
| **LLM Infrastructure** | Conversation storage, prompt management, usage tracking, billing data. |
| **Cloud Infrastructure** | Resource state storage, inventory management, audit logging. |
| **GPU ML Systems** | GPU allocation records, training job persistence, resource tracking. |
| **Distributed Systems** | State persistence, distributed lock implementation, leader election with advisory locks. |

### Practical Exercises

- [ ] Connect to PostgreSQL using `pgxpool` with proper connection pool configuration
- [ ] Implement CRUD operations using `database/sql` with prepared statements
- [ ] Implement the same CRUD using `sqlx` with struct scanning
- [ ] Implement the same using GORM with associations
- [ ] Implement a transaction: create a user and their initial settings atomically
- [ ] Use `pgx.Batch` for bulk operations
- [ ] Implement database migration with `golang-migrate`
- [ ] Benchmark `pgx` vs GORM for a read-heavy workload

### Mini Projects

- [ ] **Repository Pattern Implementation**: Build a data layer with:
  - Repository interface in the domain layer
  - pgx implementation for production
  - In-memory implementation for testing
  - Transactions that span multiple repositories
  - Connection pool monitoring
  - Query logging middleware
- [ ] **Multi-Tenant Data Layer**: Build a data layer that supports:
  - Schema-per-tenant isolation (PostgreSQL schemas)
  - Tenant-aware queries
  - Migration per tenant
  - Connection pool per tenant

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| LRU Cache | LeetCode #146 | Medium | Cache layer for database |
| Design In-Memory File System | LeetCode #588 | Hard | Data persistence abstraction |
| All O'one Data Structure | LeetCode #432 | Hard | Efficient data access patterns |

### Interview Focused Notes

**Common Interview Questions:**
- When would you use GORM vs sqlx vs pgx?
- How do you prevent SQL injection in Go?
- How do you configure a database connection pool?
- How do you handle database migrations in production?
- What is the N+1 problem and how do you solve it in Go?
- How do you implement the repository pattern in Go?

**Common Mistakes:**
- Not configuring connection pool limits (exhausting database connections)
- SQL injection via string concatenation (always use parameterized queries)
- Not closing rows (`defer rows.Close()`)
- Using GORM's AutoMigrate in production (use proper migration tools)
- Not handling `sql.ErrNoRows` (treating "not found" as a real error)

**Interviewer Expectations:**
- Chooses the right library for the use case
- Configures connection pools appropriately
- Uses parameterized queries always
- Implements proper error handling (not found vs database error)
- Tests data layer with integration tests or mocks

---

## 9.3 Redis and Caching

### Concepts to Learn

- [ ] Redis data structures: strings, lists, sets, sorted sets, hashes, streams
- [ ] Redis commands: GET, SET, DEL, EXPIRE, TTL, INCR, LPUSH, RPUSH, SADD, ZADD, HSET
- [ ] Redis as cache: cache-aside, write-through, write-behind patterns
- [ ] Cache invalidation strategies: TTL, explicit invalidation, event-driven
- [ ] Cache stampede prevention: singleflight, probabilistic early expiration
- [ ] Redis pub/sub: SUBSCRIBE, PUBLISH
- [ ] Redis streams: persistent message queue
- [ ] Redis transactions: MULTI, EXEC, WATCH (optimistic locking)
- [ ] Redis Lua scripting: atomic multi-step operations
- [ ] Redis Cluster: sharding, hash slots
- [ ] Redis Sentinel: high availability
- [ ] Go Redis client: `go-redis/redis` library
- [ ] Connection pooling with go-redis
- [ ] Distributed locking with Redis (Redlock algorithm)
- [ ] Rate limiting with Redis (sliding window, token bucket)
- [ ] Session storage with Redis
- [ ] Leaderboard with sorted sets

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Caching CI/CD pipeline results, session storage for internal tools. |
| **Platform Engineering** | Platform-wide caching layer, rate limiting, session management, distributed locking. |
| **Kubernetes** | Operators may use Redis for caching, rate limiting. Platform services behind Kubernetes use Redis heavily. |
| **MLOps** | Feature caching, model prediction caching, experiment result caching. |
| **LLM Infrastructure** | Prompt caching, KV cache for inference, response caching, rate limiting per API key. |
| **Cloud Infrastructure** | Resource state caching, API response caching, distributed locking for resource management. |
| **GPU ML Systems** | GPU allocation state caching, job queue with Redis lists/streams. |
| **Distributed Systems** | Distributed caching, pub/sub for event broadcasting, distributed locks, rate limiting. |

### Practical Exercises

- [ ] Set up Redis and connect with `go-redis`
- [ ] Implement cache-aside pattern: check cache → miss → query DB → populate cache
- [ ] Implement a rate limiter with Redis sorted sets (sliding window)
- [ ] Implement a distributed lock with Redis (SET NX EX pattern)
- [ ] Use Redis pub/sub for real-time notifications
- [ ] Build a leaderboard with sorted sets
- [ ] Use `golang.org/x/sync/singleflight` to prevent cache stampede
- [ ] Implement session storage with Redis

### Mini Projects

- [ ] **Caching Layer Library**: Build a reusable caching library with:
  - Cache-aside pattern with configurable TTL
  - Singleflight for cache stampede prevention
  - Multi-level cache (in-memory L1 + Redis L2)
  - Cache invalidation by key and by pattern
  - Cache metrics (hit rate, miss rate, eviction count)
  - Serialization (JSON, msgpack)
- [ ] **Real-Time Leaderboard**: Build a real-time leaderboard service:
  - Score submission via API
  - Top-N leaderboard with sorted sets
  - User rank lookup
  - Time-windowed leaderboards (daily, weekly)
  - Real-time updates via SSE or WebSocket

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| LRU Cache | LeetCode #146 | Medium | Cache eviction |
| LFU Cache | LeetCode #460 | Hard | Advanced cache eviction |
| Design Leaderboard | LeetCode #1244 | Medium | Sorted set concept |
| Kth Largest Element in a Stream | LeetCode #703 | Easy | Sorted set / heap |

### Interview Focused Notes

**Common Interview Questions:**
- When would you use Redis vs in-memory cache?
- What are the cache invalidation strategies?
- What is cache stampede and how do you prevent it?
- How do you implement distributed locking with Redis?
- What is the difference between Redis pub/sub and Redis streams?
- How does Redis Cluster work?

**Common Mistakes:**
- Cache without TTL (stale data forever)
- Cache stampede (all goroutines miss cache simultaneously)
- Distributed lock without expiration (lock held forever on crash)
- Using Redis for durable storage (it's primarily in-memory)
- Not handling Redis connection failures gracefully

**Interviewer Expectations:**
- Chooses appropriate caching strategy
- Prevents cache stampede
- Implements distributed locking correctly
- Handles Redis failures gracefully (fallback to database)

---

---

# STAGE 10: TESTING, BENCHMARKING, AND PROFILING

---

## 10.1 Unit Testing

### Concepts to Learn

- [ ] `testing` package: `func TestXxx(t *testing.T)`
- [ ] `t.Error`, `t.Errorf`, `t.Fatal`, `t.Fatalf`, `t.Log`, `t.Logf`
- [ ] `t.Run` for subtests: table-driven tests
- [ ] `t.Parallel()` for parallel test execution
- [ ] `t.Helper()` for marking test helpers (better error messages)
- [ ] `t.Cleanup()` for test cleanup (called after test ends)
- [ ] `t.Skip()`, `t.Skipf()` for conditional test skipping
- [ ] `t.TempDir()` for temporary directories
- [ ] `t.Setenv()` for environment variables in tests (Go 1.17+)
- [ ] Table-driven tests: the Go way
  ```go
  tests := []struct {
      name     string
      input    int
      expected int
  }{
      {"positive", 5, 25},
      {"zero", 0, 0},
      {"negative", -3, 9},
  }
  for _, tt := range tests {
      t.Run(tt.name, func(t *testing.T) { ... })
  }
  ```
- [ ] Test file naming: `*_test.go` (excluded from production builds)
- [ ] Test package naming: same package (white-box) vs `_test` suffix (black-box)
- [ ] `testify` library: `assert`, `require`, `suite`, `mock`
  - `assert.Equal`, `assert.NoError`, `assert.Contains`
  - `require.Equal` (stops test on failure)
  - `mock.Mock` for creating mock objects
- [ ] `gomock` + `mockgen`: generate mock implementations from interfaces
- [ ] Test fixtures: testdata directory, golden files
- [ ] `go test ./...`: run all tests
- [ ] `go test -v`: verbose output
- [ ] `go test -run TestName`: run specific tests
- [ ] `go test -count=1`: disable test caching
- [ ] `go test -cover`: coverage report
- [ ] `go test -coverprofile=cover.out && go tool cover -html=cover.out`: HTML coverage report
- [ ] `go test -race`: race detector
- [ ] `go test -short`: skip long tests (check `testing.Short()`)
- [ ] `TestMain(m *testing.M)`: test setup and teardown for entire package

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Testing CLI tools, testing automation scripts, CI pipeline validation. |
| **Platform Engineering** | Platform SDKs need excellent test coverage. Test-driven development for critical services. |
| **Kubernetes** | Controller tests, webhook tests, integration tests with envtest. Kubernetes has extensive test conventions. |
| **MLOps** | Pipeline logic tests, data transformation tests, model validation tests. |
| **LLM Infrastructure** | Prompt template tests, request validation tests, response parsing tests. |
| **Cloud Infrastructure** | Provider logic tests, resource CRUD tests, state management tests. |
| **GPU ML Systems** | Device allocation logic tests, scheduling algorithm tests. |
| **Distributed Systems** | Consensus algorithm tests, failure scenario tests, property-based tests. |

### Practical Exercises

- [ ] Write table-driven tests for a `Calculator` struct with Add, Subtract, Multiply, Divide
- [ ] Use `testify/assert` and `testify/require` for assertions
- [ ] Generate mocks with `mockgen` and write tests using mocked dependencies
- [ ] Write a test with `t.Parallel()` and `t.Cleanup()`
- [ ] Implement `TestMain` for package-level setup (start test database) and teardown
- [ ] Use golden files for testing complex output (JSON responses, CLI output)
- [ ] Achieve >80% coverage and generate HTML coverage report
- [ ] Run tests with `-race` flag

### Mini Projects

- [ ] **Test Suite for a Service Layer**: Given a `UserService` with Create, Get, Update, Delete, List:
  - Table-driven unit tests for each method
  - Mock repository using `testify/mock` or `gomock`
  - Error case testing
  - Edge case testing
  - 90%+ coverage
- [ ] **Golden File Testing Library**: Build a helper that compares output against golden files and updates them with a flag (`-update`)

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Valid Palindrome | LeetCode #125 | Easy | Write comprehensive test cases |
| Merge Intervals | LeetCode #56 | Medium | Complex test cases with edge cases |
| Median of Two Sorted Arrays | LeetCode #4 | Hard | Testing complex algorithms |

### Interview Focused Notes

**Common Interview Questions:**
- What are table-driven tests and why are they the Go convention?
- How do you mock dependencies in Go tests?
- What is the difference between `t.Error` and `t.Fatal`?
- How do you test code with external dependencies (database, API)?
- What is `TestMain` and when would you use it?
- How do you measure test coverage in Go?

**Common Mistakes:**
- Not using table-driven tests (repetitive test code)
- Mocking everything (testing mocks, not real code)
- Not testing error paths
- Not using `-race` flag in CI
- 100% coverage as a goal (diminishing returns; focus on critical paths)

**Interviewer Expectations:**
- Writes clean, table-driven tests
- Mocks external dependencies at the interface boundary
- Tests both happy path and error cases
- Uses race detector consistently
- Knows when to unit test vs integration test

---

## 10.2 Integration Testing and Benchmarking

### Concepts to Learn

**Integration Testing:**
- [ ] Integration tests: testing multiple components together
- [ ] `testcontainers-go`: spin up Docker containers for tests (PostgreSQL, Redis, Kafka)
- [ ] Test databases: setup, migration, seed, cleanup
- [ ] `httptest.NewServer`: integration testing HTTP servers
- [ ] `envtest` for Kubernetes controller testing (fake API server)
- [ ] Test isolation: each test gets its own state (transactions, separate databases)
- [ ] Build tags for separating unit and integration tests: `//go:build integration`
- [ ] Test ordering and dependencies: use `t.Run` for ordered subtests

**Benchmarking:**
- [ ] `func BenchmarkXxx(b *testing.B)`: benchmark function signature
- [ ] `b.N`: the runtime-determined number of iterations
- [ ] `b.ResetTimer()`: exclude setup time
- [ ] `b.StopTimer()`, `b.StartTimer()`: pause/resume timing
- [ ] `b.ReportAllocs()`: report memory allocations
- [ ] `go test -bench=. -benchmem`: run benchmarks with memory stats
- [ ] `go test -bench=. -benchtime=5s`: set benchmark duration
- [ ] `go test -bench=. -count=5`: run benchmark multiple times for stability
- [ ] `benchstat`: compare benchmark results statistically
- [ ] Sub-benchmarks: `b.Run("case", func(b *testing.B) { ... })`
- [ ] Benchmark comparison: old vs new implementation
- [ ] Avoid compiler optimization: use `b.N` result to prevent dead code elimination

**Fuzzing (Go 1.18+):**
- [ ] `func FuzzXxx(f *testing.F)`: fuzz test function
- [ ] `f.Add()`: seed corpus
- [ ] `f.Fuzz(func(t *testing.T, data []byte) { ... })`: fuzz function
- [ ] `go test -fuzz=FuzzXxx`: run fuzzer
- [ ] Crash corpus: `testdata/fuzz/` directory
- [ ] When to fuzz: parsers, validators, serializers/deserializers

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Integration tests for CI pipelines, benchmarking build tools, fuzzing input parsers. |
| **Platform Engineering** | Integration tests for multi-service interactions, benchmarking API endpoints, fuzzing API inputs. |
| **Kubernetes** | envtest for controller integration tests, benchmarking reconciliation performance. |
| **MLOps** | Integration tests for pipeline execution, benchmarking data processing. |
| **LLM Infrastructure** | Benchmarking inference latency, integration tests for serving pipeline, fuzzing input validation. |
| **Cloud Infrastructure** | Integration tests with cloud APIs (using testcontainers or mocks), benchmarking provisioning. |
| **GPU ML Systems** | Benchmarking GPU scheduling algorithms, integration tests with device plugins. |
| **Distributed Systems** | Integration tests for distributed protocols, benchmarking consensus, fuzzing protocol parsers. |

### Practical Exercises

- [ ] Write integration tests using `testcontainers-go` for PostgreSQL
- [ ] Write HTTP integration tests using `httptest.NewServer`
- [ ] Use build tags to separate unit and integration tests
- [ ] Write benchmarks comparing JSON serialization: `encoding/json` vs `json-iterator` vs `sonic`
- [ ] Use `benchstat` to compare performance of two implementations
- [ ] Write a fuzz test for a URL parser
- [ ] Benchmark slice append with pre-allocation vs without

### Mini Projects

- [ ] **Complete Test Suite**: For a REST API service, build:
  - Unit tests with mocks for service layer
  - Integration tests with testcontainers (PostgreSQL + Redis)
  - HTTP integration tests with httptest
  - Benchmarks for hot paths (serialization, database queries)
  - Fuzz tests for input parsers
  - 85%+ coverage
  - CI pipeline running all test types
- [ ] **Benchmark Lab**: Create a benchmark suite comparing:
  - JSON libraries (encoding/json, jsoniter, sonic, easyjson)
  - Logging libraries (log, zap, zerolog, slog)
  - HTTP routers (net/http, gin, chi, echo)
  - Sort implementations for different data sizes

### Interview Focused Notes

**Common Interview Questions:**
- How do you write integration tests in Go?
- What is testcontainers and how does it help?
- How do you write benchmarks in Go?
- What is fuzzing and when would you use it?
- How do you ensure test isolation?
- How do you separate unit and integration tests?

**Common Mistakes:**
- Integration tests that depend on external state (not isolated)
- Benchmarks that don't use `b.N` correctly
- Not using `b.ResetTimer()` when there's setup code
- Fuzz tests that don't exercise interesting code paths (need good seed corpus)

**Interviewer Expectations:**
- Writes both unit and integration tests
- Uses testcontainers for realistic integration tests
- Can benchmark and interpret results
- Knows fuzzing basics

---

---

# STAGE 11: GO FOR DEVOPS — CLI Tools and Automation

---

## 11.1 Building CLI Tools

### Concepts to Learn

- [ ] `flag` package: basic flag parsing (standard library)
- [ ] Cobra library: command-based CLI framework
  - Root command, sub-commands, flags (persistent, local)
  - Argument validation
  - Help text generation
  - Shell completion generation (bash, zsh, fish, PowerShell)
  - Command aliases
- [ ] Viper library: configuration management
  - Multiple sources: flags, environment variables, config files (YAML, JSON, TOML)
  - Priority: flag > env > config file > default
  - Automatic environment variable binding
  - Live config watching (file change detection)
  - Remote config (etcd, Consul)
- [ ] CLI UX best practices:
  - Exit codes: 0 (success), 1 (general error), 2 (usage error)
  - Color output: `fatih/color`, `charmbracelet/lipgloss`
  - Progress bars: `schollz/progressbar`
  - Tables: `olekukonez/tablewriter`, `charmbracelet/bubbletea`
  - Interactive prompts: `manifoldco/promptui`, `AlecAivazis/survey`
  - Spinners for long operations
- [ ] Output formats: human-readable, JSON, YAML, table (like kubectl)
- [ ] Logging in CLI tools: stderr for logs, stdout for output
- [ ] Signal handling for graceful cancellation
- [ ] Piping: reading from stdin, writing to stdout (Unix philosophy)
- [ ] Configuration file locations: `~/.config/toolname/`, XDG directories
- [ ] Auto-update mechanism for CLI tools

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Building internal CLI tools is a primary DevOps activity. kubectl, terraform, docker — all are Go CLI tools. |
| **Platform Engineering** | Platform CLI tools for developers (deploy, logs, config management). `argo`, `flux` are examples. |
| **Kubernetes** | kubectl plugins, custom controllers with CLI management, Helm-like tools. |
| **MLOps** | ML experiment CLI, model management CLI, data pipeline management CLI. |
| **LLM Infrastructure** | Model management CLI, inference configuration CLI, prompt management CLI. |
| **Cloud Infrastructure** | Cloud management CLI (like aws-cli but custom), infrastructure provisioning CLI. |
| **GPU ML Systems** | GPU management CLI, training job submission CLI. |
| **Distributed Systems** | Cluster management CLI, node administration CLI, diagnostic tools. |

### Practical Exercises

- [ ] Build a CLI with Cobra: root command + 3 sub-commands with flags
- [ ] Integrate Viper for configuration: flag > env > config file hierarchy
- [ ] Add shell completion generation
- [ ] Add table, JSON, and YAML output formats (switchable with `--output` flag)
- [ ] Add a progress bar for a long-running operation
- [ ] Read from stdin and write to stdout (Unix pipe-friendly)
- [ ] Add interactive prompts for dangerous operations (with `--yes` to skip)
- [ ] Implement `--dry-run` flag for all mutating commands

### Mini Projects

- [ ] **kubectl-like CLI**: Build a CLI tool for managing resources in a custom system:
  - Commands: `get`, `create`, `delete`, `describe`, `apply`, `logs`
  - Table output (default), JSON, YAML
  - Context/namespace support (like kubeconfig)
  - Shell completion
  - Config file support
  - Dry-run mode
- [ ] **DevOps Automation CLI**: Build a CLI tool that:
  - Manages deployments (deploy, rollback, status)
  - Manages configurations (get, set, diff)
  - Runs health checks across multiple servers
  - Generates reports (deployment history, health status)
  - Supports multiple environments (dev, staging, prod)

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Mini Parser | LeetCode #385 | Medium | Parsing CLI-like input |
| Decode String | LeetCode #394 | Medium | Parsing nested structures |
| Basic Calculator | LeetCode #224 | Hard | Expression parsing |

### Interview Focused Notes

**Common Interview Questions:**
- How do you build a CLI tool in Go?
- What is Cobra and Viper? How do they work together?
- How do you handle configuration precedence (flags > env > file)?
- How do you make a CLI tool user-friendly?
- How do you implement dry-run mode?
- How do you handle graceful cancellation in a CLI tool?

**Common Mistakes:**
- Output to stdout that isn't machine-parsable (mix of logs and data)
- Not supporting `--output json` for automation
- Not implementing `--dry-run` for dangerous operations
- Hardcoding paths instead of using XDG directories
- Not handling Ctrl+C gracefully

**Interviewer Expectations:**
- Can build a production CLI tool with Cobra/Viper
- Follows Unix philosophy (stdin/stdout, exit codes)
- Supports multiple output formats
- Handles configuration from multiple sources
- Implements safety features (dry-run, confirmation prompts)

---

## 11.2 Infrastructure Automation and CI/CD

### Concepts to Learn

- [ ] Terraform provider development in Go:
  - Provider SDK (`terraform-plugin-sdk/v2`, `terraform-plugin-framework`)
  - Resource schema definition
  - CRUD functions
  - Data sources
  - Import support
  - Testing terraform providers
- [ ] GitHub API automation:
  - `google/go-github` library
  - Repository management, PR automation, issue management
  - GitHub Actions integration
  - Webhook handling
- [ ] Docker API:
  - Docker Engine SDK (`docker/docker/client`)
  - Container management, image management
  - Docker Compose programmatic control
- [ ] CI/CD concepts:
  - Pipeline as code
  - Build, test, lint, security scan, deploy stages
  - Artifact management
  - Environment promotion
- [ ] Infrastructure as Code patterns:
  - Declarative vs imperative
  - State management
  - Drift detection
  - Idempotent operations
- [ ] GitOps principles:
  - Git as single source of truth
  - Pull-based deployment
  - Reconciliation loops (Flux, ArgoCD)

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | This IS the job. Building automation, CI/CD pipelines, infrastructure tools. |
| **Platform Engineering** | Internal developer platform tooling, self-service infrastructure, custom Terraform providers. |
| **Kubernetes** | GitOps controllers (Flux, ArgoCD written in Go), custom deployment controllers. |
| **MLOps** | ML pipeline automation, experiment tracking automation, model deployment CI/CD. |
| **LLM Infrastructure** | Model deployment automation, inference infrastructure provisioning. |
| **Cloud Infrastructure** | Custom Terraform providers, cloud automation tools, fleet management. |
| **GPU ML Systems** | GPU cluster provisioning automation, training job CI/CD. |
| **Distributed Systems** | Cluster provisioning automation, rolling update orchestration. |

### Practical Exercises

- [ ] Build a simple Terraform provider for managing a mock resource (files on disk)
- [ ] Use `go-github` to list repos, create issues, and manage PRs programmatically
- [ ] Use Docker SDK to build, run, and manage containers from Go
- [ ] Build a deployment automation tool that deploys a service via SSH
- [ ] Implement a GitOps reconciler that watches a git repo and applies changes

### Mini Projects

- [ ] **Custom Terraform Provider**: Build a Terraform provider for a custom service:
  - Manage 2-3 resource types
  - CRUD operations
  - Import existing resources
  - Data sources
  - Acceptance tests
  - Documentation generation
- [ ] **CI/CD Pipeline Tool**: Build a CI/CD tool in Go that:
  - Reads pipeline definition from YAML
  - Executes stages (build, test, lint, deploy)
  - Runs stages in Docker containers
  - Reports results
  - Supports parallel stages
  - Caches between runs
- [ ] **GitOps Controller**: Build a simple GitOps controller that:
  - Watches a git repository for changes
  - Compares desired state (in git) with actual state
  - Applies changes to reconcile
  - Reports drift

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Course Schedule | LeetCode #207 | Medium | DAG for pipeline dependencies |
| Course Schedule II | LeetCode #210 | Medium | Topological sort for execution order |
| Parallel Courses | LeetCode #1136 | Medium | Parallel execution scheduling |
| Task Scheduler | LeetCode #621 | Medium | Task scheduling with constraints |

### Interview Focused Notes

**Common Interview Questions:**
- How do you build a Terraform provider?
- What is GitOps and how does it work?
- How do you automate GitHub workflows from Go?
- How do you implement a CI/CD pipeline programmatically?
- What is infrastructure as code? What are its principles?
- How do you handle secrets in automation?

**Common Mistakes:**
- Not making operations idempotent (running twice causes issues)
- Not handling partial failures in multi-step automation
- Hardcoding credentials instead of using secret managers
- Not implementing drift detection (desired vs actual state)
- Not testing automation thoroughly (it's code, it needs tests)

**Interviewer Expectations:**
- Can build automation tools in Go
- Understands Terraform provider development
- Follows GitOps principles
- Makes operations idempotent
- Handles failures gracefully in automation

---

---

---

# STAGE 12: SECURITY IN GO

---

## 12.1 Secure Coding Practices

### Concepts to Learn

- [ ] OWASP Top 10 awareness and Go-specific mitigations
- [ ] SQL injection prevention: parameterized queries (always), never string concatenation
- [ ] XSS prevention: `html/template` auto-escaping, `text/template` pitfalls
- [ ] Command injection: `exec.Command(binary, args...)` NOT `exec.Command("sh", "-c", userInput)`
- [ ] Path traversal: `filepath.Clean`, validate paths don't escape base directory
- [ ] SSRF (Server-Side Request Forgery): validate/restrict outgoing URLs
- [ ] Insecure deserialization: validate input before unmarshaling
- [ ] Sensitive data in logs: redact passwords, tokens, PII
- [ ] Sensitive data in memory: zero out after use, avoid string (use `[]byte`)
- [ ] Cryptography:
  - `crypto/rand` for random number generation (NOT `math/rand`)
  - `crypto/sha256`, `crypto/sha512` for hashing
  - `golang.org/x/crypto/bcrypt` for password hashing
  - `crypto/aes` for encryption
  - `crypto/tls` for TLS configuration
- [ ] TLS best practices: minimum TLS 1.2, strong cipher suites, certificate validation
- [ ] `gosec` (Go security checker): static analysis for security issues
- [ ] `govulncheck`: check dependencies for known vulnerabilities
- [ ] Dependency security: `go mod tidy`, lock checksums in `go.sum`
- [ ] Container security: minimal base images, non-root user, read-only filesystem
- [ ] Secrets management: never in code, use env vars, Vault, cloud secrets managers
- [ ] Rate limiting and DoS prevention
- [ ] Input validation at every boundary

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Securing automation tools, managing secrets in CI/CD, security scanning in pipelines. |
| **Platform Engineering** | Platform API security, multi-tenant isolation, secrets management infrastructure. |
| **Kubernetes** | Pod security, network policies, RBAC, secrets management, admission webhook security. |
| **MLOps** | Model access control, training data privacy, secure model serving. |
| **LLM Infrastructure** | Prompt injection prevention, API key security, rate limiting, data privacy. |
| **Cloud Infrastructure** | IAM, credential management, encryption at rest/transit, compliance. |
| **GPU ML Systems** | GPU access control, secure multi-tenant GPU sharing. |
| **Distributed Systems** | mTLS, authentication, authorization, zero-trust networking. |

### Practical Exercises

- [ ] Run `gosec` on a project and fix all findings
- [ ] Run `govulncheck` and update vulnerable dependencies
- [ ] Implement password hashing with bcrypt
- [ ] Create TLS configuration with minimum TLS 1.2 and strong ciphers
- [ ] Implement input validation middleware
- [ ] Build a secrets manager client that reads from environment variables and Vault
- [ ] Audit a Go application for OWASP Top 10 vulnerabilities
- [ ] Build a Dockerfile following security best practices (multi-stage, non-root, minimal base)

### Mini Projects

- [ ] **Security Hardened API**: Build an API with every security measure:
  - Input validation on all endpoints
  - Parameterized queries
  - JWT auth with proper validation
  - Rate limiting
  - Security headers
  - CORS configuration
  - Request size limits
  - TLS with strong configuration
  - `gosec` passing, `govulncheck` clean
  - Structured logging with PII redaction
- [ ] **Secrets Rotation Tool**: Build a tool that:
  - Reads secrets from Vault/cloud secrets manager
  - Rotates database passwords
  - Updates application configuration
  - Verifies connectivity with new credentials
  - Rolls back on failure

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Valid Sudoku | LeetCode #36 | Medium | Input validation |
| Detect Capital | LeetCode #520 | Easy | String validation |

### Interview Focused Notes

**Common Interview Questions:**
- How do you prevent SQL injection in Go?
- How do you hash passwords in Go?
- What is the difference between `crypto/rand` and `math/rand`?
- How do you configure TLS in Go?
- How do you manage secrets in a production Go application?
- What tools do you use for Go security scanning?
- How do you prevent command injection?

**Common Mistakes:**
- Using `math/rand` for security-sensitive operations
- String concatenation in SQL queries
- Logging sensitive data (passwords, tokens, API keys)
- Not validating TLS certificates in HTTP clients (disabling verification)
- Storing secrets in code or config files committed to git
- Not running security scanners in CI

**Interviewer Expectations:**
- Security-first mindset in code
- Knows OWASP Top 10 mitigations for Go
- Uses proper cryptographic functions
- Manages secrets correctly
- Runs security tools in CI pipeline

---

## 12.2 OAuth, JWT, and Secrets Management

### Concepts to Learn

- [ ] JWT deep dive:
  - Libraries: `golang-jwt/jwt/v5`
  - Token creation, signing, validation
  - Claims: standard claims, custom claims
  - Key management: symmetric (HS256) vs asymmetric (RS256, ES256)
  - JWK (JSON Web Key) and JWKS (JWK Set) endpoints
  - Token refresh flow
  - Token revocation strategies (blacklist, short TTL)
- [ ] OAuth 2.0 implementation:
  - `golang.org/x/oauth2` package
  - Authorization code flow implementation
  - Client credentials flow for service-to-service
  - Token storage and refresh
  - Scope management
- [ ] OpenID Connect:
  - `coreos/go-oidc` library
  - ID tokens, UserInfo endpoint
  - Discovery document
  - Token validation
- [ ] Secrets management:
  - HashiCorp Vault: `hashicorp/vault/api`
  - AWS Secrets Manager, GCP Secret Manager, Azure Key Vault
  - Kubernetes Secrets (base64, not encrypted by default)
  - Sealed Secrets, External Secrets Operator
  - Secret rotation patterns
  - Dynamic secrets (Vault database secrets engine)
- [ ] mTLS (mutual TLS):
  - Client certificate authentication
  - Certificate authority (CA) management
  - Certificate rotation

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | CI/CD pipeline authentication, secret management infrastructure, automated credential rotation. |
| **Platform Engineering** | SSO integration, service-to-service auth, platform-wide secret management. |
| **Kubernetes** | Service account tokens (JWT), RBAC, Vault integration, external secrets, certificate management. |
| **MLOps** | Model serving authentication, pipeline execution credentials, data access authorization. |
| **LLM Infrastructure** | API key management, per-customer authentication, usage-based authorization. |
| **Cloud Infrastructure** | Cloud IAM integration, credential management, cross-account auth. |
| **GPU ML Systems** | GPU resource access control, training job authentication. |
| **Distributed Systems** | Service-to-service mTLS, distributed authentication, token propagation. |

### Practical Exercises

- [ ] Implement JWT creation and validation with `golang-jwt`
- [ ] Implement RS256 signing with public/private key pair
- [ ] Build OAuth 2.0 client credentials flow
- [ ] Integrate with a Vault instance to read/write secrets
- [ ] Implement mTLS between two Go services
- [ ] Build a JWKS endpoint for key rotation
- [ ] Implement token refresh with sliding expiration

### Mini Projects

- [ ] **Identity Provider**: Build a minimal identity provider:
  - User registration and login
  - JWT token issuance (RS256)
  - JWKS endpoint for public key distribution
  - Token refresh endpoint
  - OAuth 2.0 authorization code flow
  - OIDC discovery document
- [ ] **Secret Management Platform**: Build a secrets platform:
  - CRUD for secrets via API
  - Encryption at rest (AES-256)
  - Access control per secret
  - Audit logging
  - Version history
  - Dynamic secret generation (database credentials)

### Interview Focused Notes

**Common Interview Questions:**
- How does JWT authentication work end-to-end?
- What is the difference between symmetric and asymmetric JWT signing?
- How do you handle JWT token revocation?
- What is OAuth 2.0? Explain the authorization code flow.
- How do you manage secrets in Kubernetes?
- What is mTLS and when would you use it?
- How does Vault work?

**Common Mistakes:**
- Storing sensitive data in JWT payload without encryption
- Not validating JWT algorithm (algorithm confusion attack)
- Long-lived JWT tokens without refresh mechanism
- Kubernetes Secrets without encryption at rest
- Hardcoding secrets in application code
- Not rotating secrets/certificates

**Interviewer Expectations:**
- Implements JWT correctly with proper validation
- Understands OAuth 2.0 flows
- Manages secrets using proper tools (not env vars in production)
- Knows mTLS for service-to-service auth
- Has a secrets rotation strategy

---

---

# STAGE 13: CONTAINERS AND KUBERNETES FUNDAMENTALS

---

## 13.1 Docker and Containers

### Concepts to Learn

- [ ] Container concepts: namespaces, cgroups, overlay filesystem, container runtime
- [ ] Dockerfile for Go: multi-stage builds, minimal images
  ```dockerfile
  FROM golang:1.22 AS builder
  WORKDIR /app
  COPY go.mod go.sum ./
  RUN go mod download
  COPY . .
  RUN CGO_ENABLED=0 GOOS=linux go build -o /app/server ./cmd/server

  FROM gcr.io/distroless/static-debian12
  COPY --from=builder /app/server /server
  ENTRYPOINT ["/server"]
  ```
- [ ] Base images: `scratch`, `distroless`, `alpine`, `debian-slim`
- [ ] Image optimization: layer caching, .dockerignore, minimal dependencies
- [ ] Container security: non-root user, read-only filesystem, security scanning
- [ ] Docker Compose for local development
- [ ] Container registries: Docker Hub, ECR, GCR, ACR, Harbor
- [ ] Image tagging strategies: semantic versioning, git SHA, latest
- [ ] Health checks in containers
- [ ] Container resource limits: CPU, memory
- [ ] Go-specific container considerations:
  - `CGO_ENABLED=0` for static binaries
  - Scratch images for minimal footprint
  - Signal handling for graceful shutdown
  - `/etc/ssl/certs` for TLS in minimal images

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Container builds are in every CI/CD pipeline. Image optimization reduces costs and deployment time. |
| **Platform Engineering** | Container platform management, image policies, registry management. |
| **Kubernetes** | Kubernetes runs containers. Understanding container internals is fundamental. |
| **MLOps** | Training containers, serving containers, data processing containers. |
| **LLM Infrastructure** | Model serving containers (large, need optimization), inference runtime containers. |
| **Cloud Infrastructure** | Container runtime management, fleet of containerized services. |
| **GPU ML Systems** | NVIDIA container runtime, GPU-enabled containers, CUDA base images. |
| **Distributed Systems** | Containerized distributed system components, consistent deployment units. |

### Practical Exercises

- [ ] Write a multi-stage Dockerfile for a Go service
- [ ] Build with `scratch` base image, verify it works
- [ ] Build with `distroless` base image
- [ ] Compare image sizes: golang, alpine, distroless, scratch
- [ ] Set up Docker Compose with Go service + PostgreSQL + Redis
- [ ] Implement container health check
- [ ] Scan image for vulnerabilities with `trivy`
- [ ] Use Docker SDK from Go to build and run containers programmatically

### Mini Projects

- [ ] **Container Build Pipeline**: Build a Go tool that:
  - Reads Dockerfile or builds from convention
  - Builds multi-arch images (amd64, arm64)
  - Tags with git SHA and semantic version
  - Pushes to registry
  - Scans for vulnerabilities
  - Signs images with cosign

### Interview Focused Notes

**Common Interview Questions:**
- How do you optimize Docker images for Go applications?
- What is multi-stage build and why is it important for Go?
- What is the difference between `scratch`, `distroless`, and `alpine` base images?
- Why do you need `CGO_ENABLED=0` for scratch images?
- How do you handle signals in a containerized Go application?
- What are container security best practices?

**Common Mistakes:**
- Building Go images on golang base (huge image size)
- Not using multi-stage builds
- Running as root in containers
- Not setting resource limits
- Not optimizing layer caching (copying go.mod/go.sum before source code)

**Interviewer Expectations:**
- Writes optimized Dockerfiles for Go
- Knows base image trade-offs
- Follows container security best practices
- Understands container runtime concepts

---

## 13.2 Kubernetes Architecture and API

### Concepts to Learn

- [ ] Kubernetes architecture:
  - Control plane: API server, etcd, scheduler, controller manager
  - Node components: kubelet, kube-proxy, container runtime
  - API server: RESTful API, admission controllers, authentication, authorization
- [ ] Kubernetes objects: Pod, Deployment, Service, ConfigMap, Secret, Namespace
- [ ] Object model: Group, Version, Kind (GVK), Group Version Resource (GVR)
- [ ] Metadata: name, namespace, labels, annotations, UID, resourceVersion
- [ ] Spec and Status: declarative desired state (spec) and observed state (status)
- [ ] Label selectors: equality-based, set-based
- [ ] Kubernetes API conventions:
  - RESTful resources: `GET /apis/apps/v1/namespaces/default/deployments`
  - Watch API: long-polling for changes
  - Patch: strategic merge patch, JSON merge patch, JSON patch
  - Subresources: `/status`, `/scale`, `/log`
- [ ] API discovery: `/api`, `/apis`, API groups
- [ ] API versioning: `v1alpha1` → `v1beta1` → `v1`
- [ ] Kubernetes resource hierarchy: cluster-scoped vs namespace-scoped
- [ ] RBAC: Role, ClusterRole, RoleBinding, ClusterRoleBinding
- [ ] Service Accounts and token projection
- [ ] Admission controllers: mutating, validating
- [ ] Custom Resource Definitions (CRDs): extending the API
- [ ] API aggregation layer

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Managing Kubernetes clusters, deploying applications, configuring RBAC. |
| **Platform Engineering** | Building on top of Kubernetes API, extending with CRDs, building internal platforms. |
| **Kubernetes** | This IS the job. Deep understanding of every component is required. |
| **MLOps** | Kubernetes is the primary orchestration platform for ML workloads. |
| **LLM Infrastructure** | LLM serving on Kubernetes, GPU scheduling, model deployment automation. |
| **Cloud Infrastructure** | Kubernetes cluster management, multi-cluster strategies. |
| **GPU ML Systems** | GPU scheduling in Kubernetes, device plugins, resource quotas. |
| **Distributed Systems** | Kubernetes is a distributed system. Understanding its architecture teaches distributed systems principles. |

### Practical Exercises

- [ ] Set up a local Kubernetes cluster (kind, minikube, or k3d)
- [ ] Deploy a Go application with Deployment, Service, ConfigMap, Secret
- [ ] Use `kubectl` to explore the API: `kubectl api-resources`, `kubectl api-versions`
- [ ] Create RBAC rules for a service account with limited permissions
- [ ] Use `kubectl` raw API access: `kubectl get --raw /apis/apps/v1/deployments`
- [ ] Create a simple CRD and custom resource
- [ ] Watch resources using the Watch API: `kubectl get pods -w`
- [ ] Use strategic merge patch to update a deployment

### Mini Projects

- [ ] **Kubernetes Resource Explorer**: Build a Go tool that:
  - Connects to a Kubernetes cluster
  - Discovers all API resources
  - Lists resources across namespaces
  - Shows resource relationships (owner references)
  - Outputs as tree, table, or JSON

### Interview Focused Notes

**Common Interview Questions:**
- Explain Kubernetes architecture. What are the control plane components?
- What is the difference between a Deployment and a Pod?
- How does the Kubernetes API work?
- What is GVK (Group Version Kind)?
- What is the difference between Spec and Status?
- How does RBAC work in Kubernetes?
- What is a CRD?
- How do admission controllers work?

**Common Mistakes:**
- Not understanding the declarative model (telling K8s what to do vs desired state)
- Confusing Deployment, ReplicaSet, and Pod
- Not using RBAC (everything runs as cluster-admin)
- Not understanding resourceVersion (optimistic concurrency)
- Creating overly broad RBAC rules

**Interviewer Expectations:**
- Deep understanding of Kubernetes architecture
- Can explain the API model (GVK, GVR, spec/status)
- Understands RBAC and can create appropriate rules
- Knows CRDs and how to extend Kubernetes

---

## 13.3 Kubernetes Go Client (client-go)

### Concepts to Learn

- [ ] `client-go` library: the official Go client for Kubernetes
- [ ] Client types:
  - `kubernetes.Clientset`: typed client for built-in resources
  - `dynamic.Interface`: dynamic client for any resource
  - `rest.RESTClient`: raw REST client
- [ ] Authentication:
  - In-cluster: service account token (automatic)
  - Out-of-cluster: kubeconfig file
  - `rest.InClusterConfig()` vs `clientcmd.BuildConfigFromFlags()`
- [ ] CRUD operations:
  - `client.CoreV1().Pods(namespace).Get(ctx, name, opts)`
  - `client.CoreV1().Pods(namespace).List(ctx, opts)`
  - `client.CoreV1().Pods(namespace).Create(ctx, pod, opts)`
  - `client.CoreV1().Pods(namespace).Update(ctx, pod, opts)`
  - `client.CoreV1().Pods(namespace).Delete(ctx, name, opts)`
- [ ] Watch API: `client.CoreV1().Pods(namespace).Watch(ctx, opts)`
- [ ] Informers: cached, event-driven resource watching
  - `SharedInformerFactory`: creates informers for multiple resource types
  - Informer cache: reduces API server load
  - Event handlers: `AddFunc`, `UpdateFunc`, `DeleteFunc`
  - Lister: read from cache, not API server
- [ ] Work queues: `client-go/util/workqueue`
  - Rate-limited queue
  - Delayed queue
  - Deduplication
- [ ] Patch operations: strategic merge, JSON merge, apply
- [ ] Server-Side Apply: `fieldManager` based
- [ ] Pagination: `Continue` token, `Limit`
- [ ] Label and field selectors
- [ ] Dynamic client for CRDs and unstructured resources
- [ ] `apimachinery`: resource types, scheme, serialization

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Building Kubernetes automation tools, deployment scripts, health checkers. |
| **Platform Engineering** | Platform controllers, resource managers, internal Kubernetes tooling. |
| **Kubernetes** | This is the foundation for building any Kubernetes extension. |
| **MLOps** | Custom schedulers, training job managers, model deployment controllers. |
| **LLM Infrastructure** | Model deployment automation, inference scaling controllers. |
| **Cloud Infrastructure** | Multi-cluster management tools, fleet controllers. |
| **GPU ML Systems** | GPU device plugin, custom scheduler, resource tracker. |
| **Distributed Systems** | Kubernetes as a distributed systems framework, leveraging its primitives. |

### Practical Exercises

- [ ] Connect to a Kubernetes cluster from Go (both in-cluster and out-of-cluster)
- [ ] List all pods in a namespace with label filtering
- [ ] Create a deployment programmatically
- [ ] Watch for pod events and log them
- [ ] Set up a SharedInformerFactory and use Listers to read from cache
- [ ] Add event handlers (Add, Update, Delete) to an informer
- [ ] Use the work queue to process events
- [ ] Use the dynamic client to work with a CRD
- [ ] Implement pagination for listing large numbers of resources

### Mini Projects

- [ ] **Kubernetes Resource Syncer**: Build a tool that:
  - Watches configmaps in one namespace
  - Syncs them to another namespace (or cluster)
  - Handles creates, updates, and deletes
  - Uses informers and work queue
  - Handles conflicts with resourceVersion
- [ ] **Pod Lifecycle Monitor**: Build a tool that:
  - Watches all pods across all namespaces
  - Tracks pod lifecycle events (Pending → Running → Succeeded/Failed)
  - Calculates statistics (average startup time, failure rate)
  - Alerts on pods stuck in Pending for too long
  - Exposes metrics via Prometheus endpoint

### Interview Focused Notes

**Common Interview Questions:**
- What is client-go and what are the different client types?
- How do informers work? Why are they better than direct API calls?
- What is a work queue and why is it needed?
- How do you authenticate to a Kubernetes cluster from Go?
- What is the difference between typed and dynamic clients?
- What is Server-Side Apply?
- How do you handle optimistic concurrency with resourceVersion?

**Common Mistakes:**
- Making API calls in a loop instead of using informers (API server overload)
- Not handling resourceVersion conflicts (lost updates)
- Not using rate-limited work queues (overwhelming the API server)
- Forgetting to start informers (`factory.Start(stopCh)`)
- Not handling cache sync (`cache.WaitForCacheSync`)

**Interviewer Expectations:**
- Can use client-go to interact with the Kubernetes API
- Understands informer architecture and why it exists
- Uses work queues for event processing
- Handles optimistic concurrency correctly
- Can work with both typed and dynamic clients

---

---

# STAGE 14: KUBERNETES CONTROLLERS AND OPERATORS

---

## 14.1 Controller Pattern

### Concepts to Learn

- [ ] Controller definition: a loop that watches resources and reconciles actual state with desired state
- [ ] Reconciliation loop: observe → diff → act
- [ ] Level-triggered vs edge-triggered: Kubernetes controllers are level-triggered (react to state, not events)
- [ ] Controller components:
  - Informer: watches resources, caches, generates events
  - Work queue: deduplicates and rate-limits events
  - Reconciler: processes each event (the business logic)
- [ ] Owner references: parent-child relationships between resources
- [ ] Garbage collection: automatic deletion of owned resources when owner is deleted
- [ ] Finalizers: pre-deletion hooks for cleanup
- [ ] Status subresource: reporting observed state
- [ ] Conditions: standardized status reporting (`Type`, `Status`, `Reason`, `Message`, `LastTransitionTime`)
- [ ] Events: recording events on resources (`recorder.Event(obj, corev1.EventTypeNormal, "Reconciled", "...")`)
- [ ] Requeue: `ctrl.Result{Requeue: true}`, `ctrl.Result{RequeueAfter: time.Minute}`
- [ ] Error handling in controllers: requeue on error, exponential backoff
- [ ] Idempotent reconciliation: running reconcile multiple times should produce the same result
- [ ] Leader election: only one controller instance is active at a time
- [ ] Controller best practices:
  - Don't watch resources you don't own
  - Don't modify resources you don't own (except status)
  - Use server-side apply for creating/updating owned resources
  - Always set owner references on created resources
  - Use finalizers for external cleanup

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Understanding how Kubernetes controllers work helps debug deployment issues. |
| **Platform Engineering** | Building custom controllers is the primary way to extend Kubernetes platforms. |
| **Kubernetes** | Controllers ARE Kubernetes. Every built-in resource has a controller. |
| **MLOps** | Training job controllers, model serving controllers, data pipeline controllers. |
| **LLM Infrastructure** | Model deployment controllers, inference scaling controllers, prompt pipeline controllers. |
| **Cloud Infrastructure** | Cloud resource controllers (provision AWS resources from Kubernetes), multi-cloud controllers. |
| **GPU ML Systems** | GPU allocation controllers, training scheduler controllers. |
| **Distributed Systems** | The controller pattern IS a distributed systems pattern (reconciliation loops). |

### Practical Exercises

- [ ] Build a controller from scratch using client-go (no framework):
  - Watch a ConfigMap
  - On change, create/update a Deployment
  - Set owner reference
  - Implement finalizer for cleanup
- [ ] Implement status conditions on a custom resource
- [ ] Implement leader election for your controller
- [ ] Test your controller with unit tests (mock the client)
- [ ] Handle edge cases: resource already exists, resource deleted during reconcile, conflict

### Mini Projects

- [ ] **Namespace Controller**: Build a controller that watches Namespaces and automatically:
  - Creates default ResourceQuota
  - Creates default NetworkPolicy
  - Creates default LimitRange
  - Adds standard labels
  - Reports status conditions
  - Handles namespace deletion (finalizer for cleanup)

### Interview Focused Notes

**Common Interview Questions:**
- What is a Kubernetes controller?
- Explain the reconciliation loop.
- What is the difference between level-triggered and edge-triggered controllers?
- What are owner references and why are they important?
- What are finalizers and when do you use them?
- How do you handle errors in a controller?
- What is leader election and why is it needed?
- What does idempotent reconciliation mean?

**Common Mistakes:**
- Non-idempotent reconciliation (reconcile creates duplicate resources)
- Not setting owner references (orphaned resources)
- Not using finalizers for external cleanup
- Reconciling too frequently (not using backoff)
- Not reporting status (operator is a black box)
- Modifying resources you don't own

**Interviewer Expectations:**
- Can explain the controller pattern clearly
- Understands reconciliation, owner references, finalizers
- Can implement a controller from scratch
- Writes idempotent reconciliation logic
- Handles errors with proper requeue strategy

---

## 14.2 Custom Resource Definitions (CRDs)

### Concepts to Learn

- [ ] CRD definition: extending the Kubernetes API with custom resource types
- [ ] CRD specification:
  - Group, Version, Names (kind, plural, singular, short names)
  - Schema: OpenAPI v3 validation
  - Scope: Cluster or Namespaced
  - Status subresource
  - Scale subresource
  - Additional printer columns
  - Categories
- [ ] CRD versioning: multiple versions, conversion webhooks
- [ ] CRD validation: OpenAPI schema, cel-validation (Go 1.22+)
- [ ] CRD defaulting: mutating admission webhooks, CEL-based defaults
- [ ] Go type definition for CRDs:
  ```go
  type MyResource struct {
      metav1.TypeMeta   `json:",inline"`
      metav1.ObjectMeta `json:"metadata,omitempty"`
      Spec   MyResourceSpec   `json:"spec,omitempty"`
      Status MyResourceStatus `json:"status,omitempty"`
  }
  type MyResourceSpec struct { ... }
  type MyResourceStatus struct {
      Conditions []metav1.Condition `json:"conditions,omitempty"`
  }
  ```
- [ ] Code generation for CRDs:
  - `controller-gen`: generate CRD manifests from Go types
  - `deepcopy-gen`: generate DeepCopy methods
  - `register-gen`: generate scheme registration
  - `informer-gen`, `lister-gen`, `client-gen` for typed clients
- [ ] CRD markers (kubebuilder annotations):
  - `// +kubebuilder:object:root=true`
  - `// +kubebuilder:subresource:status`
  - `// +kubebuilder:validation:Required`
  - `// +kubebuilder:printcolumn`
  - `// +kubebuilder:default`

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Understanding CRDs to troubleshoot operators and platform extensions. |
| **Platform Engineering** | Defining platform API surfaces as CRDs. Custom resource types for tenants, environments, applications. |
| **Kubernetes** | CRDs are the primary extension mechanism. Every operator uses CRDs. |
| **MLOps** | CRDs for TrainingJob, Model, Pipeline, Experiment (see Kubeflow). |
| **LLM Infrastructure** | CRDs for InferenceService, ModelDeployment, PromptTemplate. |
| **Cloud Infrastructure** | CRDs for cloud resources (AWS Controller for Kubernetes uses CRDs). |
| **GPU ML Systems** | CRDs for GPUAllocation, TrainingCluster, DevicePool. |
| **Distributed Systems** | CRDs for custom distributed system components. |

### Practical Exercises

- [ ] Define a CRD in Go types with spec, status, and conditions
- [ ] Use `controller-gen` to generate CRD manifests
- [ ] Apply the CRD to a cluster and create custom resources
- [ ] Add OpenAPI validation to the CRD (required fields, enums, patterns)
- [ ] Add additional printer columns
- [ ] Add a status subresource
- [ ] Generate typed client, informer, and lister

### Mini Projects

- [ ] **Application CRD**: Design and implement a CRD for `Application`:
  - Spec: name, image, replicas, env vars, ports, health check, resources
  - Status: conditions (Available, Progressing, Degraded), readyReplicas, observedGeneration
  - Printer columns: name, replicas, status, age
  - Validation: image required, replicas >= 0, port range validation
  - Defaulting: replicas default to 1
  - Controller that creates Deployment + Service from Application spec

### Interview Focused Notes

**Common Interview Questions:**
- What is a CRD and why would you create one?
- How do you define validation on a CRD?
- What is the status subresource?
- How do CRD versions work?
- What are the code generation tools for CRDs?
- How do you handle CRD schema migration?

**Common Mistakes:**
- No validation on CRDs (garbage data in the cluster)
- Not using status subresource (status changes trigger full reconciliation)
- Not using conditions for status reporting
- Breaking backward compatibility in CRD schema
- Not setting `observedGeneration` in status

**Interviewer Expectations:**
- Can define CRD types in Go
- Understands validation, defaulting, and versioning
- Uses code generation correctly
- Follows Kubernetes API conventions (spec/status, conditions)

---

## 14.3 Kubebuilder and Operator SDK

### Concepts to Learn

- [ ] **Kubebuilder**: framework for building Kubernetes controllers
  - `kubebuilder init`: scaffold a project
  - `kubebuilder create api`: create a CRD + controller
  - `kubebuilder create webhook`: create admission webhooks
  - Project layout: `api/`, `controllers/`, `config/`, `cmd/`
  - `controller-runtime` library (used by Kubebuilder):
    - `Manager`: manages controllers, caches, clients
    - `Controller`: watches resources, enqueues reconcile requests
    - `Reconciler`: `Reconcile(ctx, Request) (Result, error)`
    - `Client`: read/write Kubernetes resources
    - `Cache`: in-memory cache of watched resources
    - `Predicate`: filter events before they reach the reconciler
    - `EventHandler`: map events to reconcile requests
    - `Source`: watch sources (Kind, Channel, Informer)
  - RBAC generation from markers
  - Webhook types: mutating, validating, conversion
- [ ] **Operator SDK**: Red Hat's framework (builds on Kubebuilder)
  - Additional features: OLM integration, scorecard, Ansible/Helm operators
  - Operator Lifecycle Manager (OLM): install, update, manage operators
  - Operator bundle format
  - OperatorHub publishing
- [ ] Operator patterns:
  - Level 1: Basic install (automated deployment)
  - Level 2: Seamless upgrades
  - Level 3: Full lifecycle (backup, restore)
  - Level 4: Deep insights (metrics, alerts)
  - Level 5: Auto-pilot (auto-scaling, auto-healing, auto-tuning)
- [ ] Testing operators:
  - `envtest`: fake API server for integration tests
  - Unit tests with fake client
  - End-to-end tests with kind cluster

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Operating and debugging operators in production. Understanding operator capabilities. |
| **Platform Engineering** | Building platform operators is THE way to build Kubernetes platforms. |
| **Kubernetes** | Kubebuilder/Operator SDK is the standard for building controllers. |
| **MLOps** | Kubeflow operators, training job operators, model serving operators. |
| **LLM Infrastructure** | Custom operators for LLM deployment, inference scaling, model management. |
| **Cloud Infrastructure** | Cloud resource operators (ACK, Crossplane use operator pattern). |
| **GPU ML Systems** | GPU operator, custom scheduling operator, resource management operator. |
| **Distributed Systems** | Database operators (postgres-operator, cassandra-operator), message queue operators. |

### Practical Exercises

- [ ] Scaffold a project with `kubebuilder init`
- [ ] Create a CRD and controller with `kubebuilder create api`
- [ ] Implement the reconcile function
- [ ] Add a validating webhook
- [ ] Add a mutating webhook (defaulting)
- [ ] Write unit tests with fake client
- [ ] Write integration tests with envtest
- [ ] Deploy the operator to a kind cluster
- [ ] Add RBAC markers and generate RBAC manifests
- [ ] Add metrics and events

### Mini Projects

- [ ] **Database Operator**: Build a Kubernetes operator that manages PostgreSQL instances:
  - CRD: `PostgreSQL` with spec (version, storage, replicas, backup schedule)
  - Controller creates: StatefulSet, Service, PVC, ConfigMap, Secret (credentials)
  - Status: conditions (Ready, BackupComplete), connectionString, version
  - Handles scaling (replicas change)
  - Handles version upgrade
  - Backup CronJob creation
  - Credentials rotation
  - Finalizer for cleanup
  - Validating webhook (valid PostgreSQL versions)
  - Mutating webhook (defaults)
  - envtest integration tests
- [ ] **Application Platform Operator**: Build an operator for a developer platform:
  - CRD: `Application` (image, replicas, env, ports, ingress, autoscaling)
  - Creates: Deployment, Service, Ingress, HPA, PDB
  - Status: URL, health, conditions
  - Supports canary deployments
  - Supports rollback

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Course Schedule | LeetCode #207 | Medium | Dependency resolution (like CRD dependencies) |
| Clone Graph | LeetCode #133 | Medium | Deep copy (like DeepCopy in CRDs) |
| Number of Islands | LeetCode #200 | Medium | Graph traversal (like resource graph) |

### Interview Focused Notes

**Common Interview Questions:**
- What is Kubebuilder and how do you use it?
- Walk me through building a Kubernetes operator.
- What is controller-runtime? What are its key components?
- How do you test a Kubernetes operator?
- What are admission webhooks? When would you use mutating vs validating?
- What is envtest?
- How do you handle upgrades in an operator?
- What are the operator maturity levels?

**Common Mistakes:**
- Reconciling based on events instead of state (not level-triggered)
- Not handling partial failures (resource A created but resource B failed)
- No finalizers for external resource cleanup
- Not testing with envtest (only unit tests with fake client)
- Not implementing leader election
- Not setting proper RBAC (over-permissive)

**Interviewer Expectations:**
- Can build an operator from scratch with Kubebuilder
- Writes idempotent reconciliation
- Tests with envtest
- Implements webhooks for validation and defaulting
- Follows operator best practices
- Can explain the controller-runtime architecture

---

---

# STAGE 15: OBSERVABILITY AND INSTRUMENTATION

---

## 15.1 Structured Logging

### Concepts to Learn

- [ ] Structured logging: key-value pairs, JSON format
- [ ] Log levels: DEBUG, INFO, WARN, ERROR, FATAL
- [ ] Standard library `log` package: basic, unstructured
- [ ] `log/slog` (Go 1.21+): structured logging in standard library
  - `slog.Info("msg", "key", value)`
  - `slog.With("key", value)` for adding context
  - Handlers: `TextHandler`, `JSONHandler`
  - Custom handlers
  - Groups for nested keys
  - LogValuer interface for lazy evaluation
- [ ] `uber-go/zap`:
  - `zap.NewProduction()`, `zap.NewDevelopment()`
  - `logger.Info("msg", zap.String("key", "value"))`
  - SugaredLogger: `sugar.Infow("msg", "key", value)`
  - Performance: zero-allocation logging
  - Field types: `zap.String`, `zap.Int`, `zap.Error`, `zap.Duration`
- [ ] `sirupsen/logrus`: popular but slower than zap
- [ ] `rs/zerolog`: zero-allocation structured logger
- [ ] Logging best practices:
  - Log at the right level (don't log everything as INFO)
  - Include context: request ID, user ID, trace ID
  - Don't log sensitive data (passwords, tokens, PII)
  - Don't log in hot loops (performance impact)
  - Use structured fields, not string formatting
  - Log once per event (don't log AND return error)
- [ ] Logging in Kubernetes: stdout/stderr, container logs, log aggregation
- [ ] Log aggregation: Fluentd, Fluent Bit, Loki, ELK stack

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Log aggregation, log-based alerting, debugging production issues. |
| **Platform Engineering** | Centralized logging infrastructure, log pipeline, log retention policies. |
| **Kubernetes** | Container logs, controller logs, audit logs. Structured logging is essential for log aggregation. |
| **MLOps** | Training logs, pipeline execution logs, model serving request logs. |
| **LLM Infrastructure** | Inference request/response logging, token usage logging, error logging. |
| **Cloud Infrastructure** | Infrastructure operation logs, audit trail, compliance logging. |
| **GPU ML Systems** | GPU operation logs, training progress logs, device error logs. |
| **Distributed Systems** | Correlated logging across services, distributed debugging, log-based observability. |

### Practical Exercises

- [ ] Set up `slog` with JSON handler
- [ ] Set up `zap` with production configuration
- [ ] Add request ID to all log lines using context
- [ ] Create a logging middleware for HTTP that logs request/response
- [ ] Implement log level changing at runtime (via HTTP endpoint or signal)
- [ ] Compare performance: `slog` vs `zap` vs `zerolog` with benchmarks

### Mini Projects

- [ ] **Logging Library Wrapper**: Build a logging library that:
  - Abstracts over `slog`/`zap` (so you can switch)
  - Adds context from `context.Context` (request ID, trace ID)
  - Redacts sensitive fields
  - Supports sampling (log every Nth message for high-volume paths)
  - Configurable via environment variables

### Interview Focused Notes

**Common Interview Questions:**
- What is structured logging and why is it important?
- Compare `slog`, `zap`, `zerolog`, and `logrus`.
- How do you correlate logs across microservices?
- What should you NOT log?
- How do you handle logging in Kubernetes?

**Common Mistakes:**
- Using `fmt.Println` for logging in production
- Logging sensitive data
- Logging at wrong levels (everything as INFO)
- String formatting instead of structured fields (not searchable)
- Logging and returning the same error (double logging)

**Interviewer Expectations:**
- Uses structured logging with appropriate library
- Includes context (request ID, trace ID) in logs
- Redacts sensitive information
- Chooses appropriate log levels

---

## 15.2 Metrics with Prometheus

### Concepts to Learn

- [ ] Prometheus data model: metric name, labels, timestamp, value
- [ ] Metric types:
  - Counter: monotonically increasing (requests_total, errors_total)
  - Gauge: can go up and down (temperature, queue_size, goroutine_count)
  - Histogram: distribution of values (request_duration_seconds)
  - Summary: similar to histogram, calculates percentiles client-side
- [ ] `prometheus/client_golang`: official Go client
  - `promauto.NewCounter`, `promauto.NewGauge`, `promauto.NewHistogram`
  - `promhttp.Handler()`: expose metrics endpoint
  - Custom collectors
  - Registries
- [ ] Metric naming conventions:
  - `<namespace>_<subsystem>_<name>_<unit>`
  - Example: `http_requests_total`, `http_request_duration_seconds`
  - Use `_total` suffix for counters
  - Use base units (seconds, bytes, not milliseconds or kilobytes)
- [ ] Labels: adding dimensions to metrics
  - Low cardinality (method, status_code, NOT user_id)
- [ ] RED method: Rate, Errors, Duration (for services)
- [ ] USE method: Utilization, Saturation, Errors (for resources)
- [ ] Four Golden Signals: latency, traffic, errors, saturation
- [ ] Go runtime metrics: `go_*` prefix, goroutines, GC, memory
- [ ] Custom business metrics: queue depth, processing lag, cache hit rate
- [ ] PromQL basics for querying metrics

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Monitoring dashboards, alerting rules, SLA/SLO tracking. |
| **Platform Engineering** | Platform metrics, tenant usage metrics, infrastructure health. |
| **Kubernetes** | Controller metrics (reconciliation duration, queue depth, errors), kube-state-metrics. |
| **MLOps** | Training metrics (loss, accuracy, GPU utilization), pipeline metrics. |
| **LLM Infrastructure** | Inference metrics (latency, tokens/sec, queue depth), model performance metrics. |
| **Cloud Infrastructure** | Infrastructure metrics, resource utilization, cost metrics. |
| **GPU ML Systems** | GPU utilization, memory usage, temperature, power consumption metrics. |
| **Distributed Systems** | Consensus metrics, replication lag, network metrics. |

### Practical Exercises

- [ ] Expose Prometheus metrics endpoint from a Go HTTP server
- [ ] Add counter for HTTP requests (labeled by method, path, status_code)
- [ ] Add histogram for request duration
- [ ] Add gauge for active goroutines and connections
- [ ] Create custom business metrics (cache hit rate, queue depth)
- [ ] Set up Prometheus to scrape your application
- [ ] Create Grafana dashboards for your metrics
- [ ] Write PromQL queries for common patterns (error rate, p99 latency)

### Mini Projects

- [ ] **Metrics Library**: Build a metrics middleware package:
  - HTTP middleware: request count, duration, response size
  - gRPC interceptor: request count, duration, status code
  - Database metrics: query count, duration, error rate
  - Cache metrics: hit rate, miss rate, eviction count
  - Custom business metrics registration
  - Pre-built Grafana dashboard JSON

### Interview Focused Notes

**Common Interview Questions:**
- What are the Prometheus metric types? When do you use each?
- What are the Four Golden Signals?
- What is the RED method?
- How do you choose labels for metrics?
- What is high cardinality and why is it a problem?
- How do you calculate error rate in PromQL?
- How do you calculate p99 latency?

**Common Mistakes:**
- High-cardinality labels (user ID, request ID as labels)
- Using gauges for things that should be counters
- Not using histograms for latency (only tracking average)
- Too many metrics (noise, performance impact)
- Not labeling metrics consistently

**Interviewer Expectations:**
- Instruments applications with appropriate metrics
- Knows metric types and when to use each
- Follows naming conventions
- Avoids high cardinality
- Can write basic PromQL

---

## 15.3 Distributed Tracing with OpenTelemetry

### Concepts to Learn

- [ ] Distributed tracing concepts: trace, span, context propagation
- [ ] OpenTelemetry (OTel): unified observability framework (traces, metrics, logs)
- [ ] OTel Go SDK:
  - Tracer provider, span processor, exporter
  - Span creation: `tracer.Start(ctx, "operation-name")`
  - Span attributes: `span.SetAttributes(attribute.String("key", "value"))`
  - Span events: `span.AddEvent("event-name")`
  - Span status: `span.SetStatus(codes.Error, "error message")`
  - Context propagation: `otel.GetTextMapPropagator()`
- [ ] Exporters: Jaeger, Zipkin, OTLP (standard)
- [ ] W3C Trace Context: standard propagation format
- [ ] Instrumentation:
  - HTTP: `otelhttp` middleware
  - gRPC: `otelgrpc` interceptors
  - Database: `otelsql`
  - Manual instrumentation for custom operations
- [ ] Sampling: always-on, probability, tail-based
- [ ] Baggage: key-value pairs propagated across services
- [ ] OTel Collector: receive, process, export telemetry data
- [ ] Trace analysis: identifying bottlenecks, error propagation, latency breakdown

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Debugging production issues across services, identifying performance bottlenecks. |
| **Platform Engineering** | Platform-wide tracing infrastructure, trace-based SLOs. |
| **Kubernetes** | Tracing through admission webhooks, controllers, API server. |
| **MLOps** | Tracing pipeline execution, identifying slow stages. |
| **LLM Infrastructure** | Tracing inference requests through preprocessing → inference → postprocessing. |
| **Cloud Infrastructure** | Tracing multi-cloud operations, cross-region request tracing. |
| **GPU ML Systems** | Tracing GPU operations, identifying scheduling bottlenecks. |
| **Distributed Systems** | Tracing is essential for understanding distributed system behavior. |

### Practical Exercises

- [ ] Set up OTel tracing in a Go HTTP server
- [ ] Add tracing to HTTP client calls
- [ ] Propagate trace context between two services
- [ ] Add span attributes and events
- [ ] Set up Jaeger and view traces
- [ ] Instrument a database query with tracing
- [ ] Build a multi-service application with end-to-end tracing

### Mini Projects

- [ ] **Observability Platform**: Build a complete observability setup:
  - Three Go microservices with inter-service calls
  - Structured logging with trace ID correlation
  - Prometheus metrics with RED method
  - OpenTelemetry distributed tracing
  - Grafana dashboards for all three signals
  - Jaeger for trace visualization
  - Alerting rules for SLO violations

### Interview Focused Notes

**Common Interview Questions:**
- What is distributed tracing and why is it important?
- What is OpenTelemetry?
- How does context propagation work?
- What is the difference between tracing, metrics, and logs?
- How do you correlate logs with traces?
- What is sampling and why is it needed?

**Common Mistakes:**
- Not propagating context (broken traces)
- Tracing everything without sampling (too much data)
- Not correlating traces with logs (missing trace ID in log entries)
- Over-instrumenting (performance overhead)

**Interviewer Expectations:**
- Can set up distributed tracing with OTel
- Understands context propagation
- Correlates traces with logs and metrics
- Knows sampling strategies

---

---

---

# STAGE 16: DISTRIBUTED SYSTEMS

---

## 16.1 Distributed Systems Fundamentals

### Concepts to Learn

- [ ] What makes a system distributed: multiple nodes, network communication, partial failures
- [ ] Fallacies of distributed computing (Peter Deutsch):
  1. The network is reliable
  2. Latency is zero
  3. Bandwidth is infinite
  4. The network is secure
  5. Topology doesn't change
  6. There is one administrator
  7. Transport cost is zero
  8. The network is homogeneous
- [ ] CAP theorem: Consistency, Availability, Partition tolerance — pick two
  - CP systems: consistent but may be unavailable during partitions (etcd, ZooKeeper)
  - AP systems: available but may be inconsistent during partitions (Cassandra, DynamoDB)
  - CA doesn't exist in distributed systems (partitions always happen)
- [ ] PACELC theorem: extension of CAP (when partitioned: AP or CP; else: latency or consistency)
- [ ] Consistency models:
  - Strong consistency: all reads see the latest write
  - Eventual consistency: all reads will eventually see the latest write
  - Causal consistency: causally related operations are seen in order
  - Read-your-writes consistency
  - Linearizability: strongest form of consistency
  - Sequential consistency
- [ ] Consensus algorithms:
  - Raft: leader election, log replication, safety (used by etcd, Consul)
  - Paxos: theoretical foundation (used by Google Chubby)
  - Raft in Go: `hashicorp/raft` library
- [ ] Time in distributed systems:
  - Wall clock vs logical clock
  - Lamport timestamps
  - Vector clocks
  - Hybrid logical clocks (HLC)
  - NTP and time synchronization challenges
- [ ] Failure modes:
  - Crash failures, Byzantine failures
  - Network partitions, split brain
  - Partial failures
- [ ] Replication strategies:
  - Single-leader replication
  - Multi-leader replication
  - Leaderless replication (quorum reads/writes)
- [ ] Sharding/partitioning:
  - Hash-based partitioning
  - Range-based partitioning
  - Consistent hashing
- [ ] Distributed transactions:
  - Two-phase commit (2PC)
  - Three-phase commit (3PC)
  - Saga pattern (choreography vs orchestration)
- [ ] Crdt (Conflict-free Replicated Data Types)
- [ ] Gossip protocols: epidemic-style information dissemination
- [ ] Service mesh and sidecar pattern

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Operating distributed systems, understanding failure modes, debugging distributed issues. |
| **Platform Engineering** | Building distributed platform services, choosing consistency models, implementing coordination. |
| **Kubernetes** | Kubernetes IS a distributed system. etcd uses Raft. Understanding distributed systems helps debug cluster issues. |
| **MLOps** | Distributed training, distributed data processing, consistent model versioning. |
| **LLM Infrastructure** | Distributed inference, model sharding, consistent model serving across replicas. |
| **Cloud Infrastructure** | Multi-region deployments, cross-region replication, global load balancing. |
| **GPU ML Systems** | Distributed GPU training (data parallelism, model parallelism), GPU cluster management. |
| **Distributed Systems** | This IS the domain. Every concept here is fundamental. |

### Practical Exercises

- [ ] Implement a simple Raft leader election using `hashicorp/raft`
- [ ] Build a replicated key-value store using Raft
- [ ] Implement consistent hashing with virtual nodes
- [ ] Build a gossip protocol for membership management
- [ ] Implement Lamport timestamps in a message passing system
- [ ] Demonstrate the split-brain problem and implement fencing tokens
- [ ] Build a distributed counter using CRDTs (G-Counter, PN-Counter)

### Mini Projects

- [ ] **Distributed Key-Value Store**: Build a distributed KV store with:
  - Raft consensus for replication
  - HTTP API for CRUD
  - Leader forwarding
  - Snapshot and recovery
  - Membership changes (add/remove nodes)
  - Linearizable reads
- [ ] **Gossip-Based Cluster Membership**: Build a cluster membership system:
  - Node join/leave detection
  - Failure detection via heartbeats
  - Metadata propagation via gossip
  - Consistent hash ring for data placement
  - Rebalancing on membership changes

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Consistent Hashing | System Design | Hard | Core distributed systems concept |
| Design Distributed Cache | System Design | Hard | Replication, consistency, partitioning |
| Design Key-Value Store | System Design | Hard | Consensus, replication |
| Design Distributed Lock | System Design | Medium | Coordination, fencing |
| Detect Cycle in Directed Graph | LeetCode #207 | Medium | Detecting deadlocks |

### Interview Focused Notes

**Common Interview Questions:**
- Explain the CAP theorem. Give examples of CP and AP systems.
- What is Raft? How does leader election work?
- What is consistent hashing? Why are virtual nodes needed?
- What are the consistency models? Explain eventual consistency.
- How do you handle distributed transactions?
- What is the saga pattern?
- What is split brain and how do you prevent it?
- How does gossip protocol work?

**Common Mistakes:**
- Thinking CAP means you choose two and ignore one (partition tolerance isn't optional)
- Using distributed transactions when saga would suffice
- Not handling network partitions gracefully
- Assuming clocks are synchronized across nodes
- Not implementing proper failure detection (false positives)

**Interviewer Expectations:**
- Can explain CAP theorem and consistency models
- Understands Raft at a practical level
- Knows consistent hashing and when to use it
- Can design for failure in distributed systems
- Understands the trade-offs between consistency and availability

---

## 16.2 Message Queues and Event-Driven Systems

### Concepts to Learn

- [ ] Message queue concepts: producer, consumer, broker, topic, partition, offset
- [ ] Delivery guarantees: at-most-once, at-least-once, exactly-once
- [ ] Message ordering: per-partition ordering, global ordering
- [ ] Consumer groups: parallel consumption, partition assignment
- [ ] Dead letter queues: handling failed messages
- [ ] Back pressure: handling slow consumers

**Apache Kafka:**
- [ ] Kafka architecture: brokers, topics, partitions, replicas, ZooKeeper/KRaft
- [ ] Producer: `confluent-kafka-go` or `segmentio/kafka-go` or `IBM/sarama`
- [ ] Consumer: consumer groups, offset management, rebalancing
- [ ] Kafka Streams concepts (implemented in Go manually)
- [ ] Kafka Connect concepts
- [ ] Schema Registry for Avro/Protobuf schemas

**NATS:**
- [ ] NATS core: pub/sub, request/reply, queue groups
- [ ] NATS JetStream: persistence, exactly-once, streams, consumers
- [ ] NATS Go client: `nats-io/nats.go`
- [ ] Use cases: microservice communication, event-driven architecture

**RabbitMQ:**
- [ ] AMQP protocol
- [ ] Exchanges, queues, bindings, routing keys
- [ ] Exchange types: direct, topic, fanout, headers
- [ ] Go client: `streadway/amqp` or `rabbitmq/amqp091-go`

- [ ] Event-driven architecture:
  - Event sourcing: storing events as the source of truth
  - CQRS: separate read/write models
  - Event-driven microservices
  - Choreography vs orchestration
  - Outbox pattern for reliable event publishing
  - Idempotent event handling

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Setting up and managing message brokers, monitoring queues, debugging message flow. |
| **Platform Engineering** | Event-driven platform architecture, async communication between services, event streaming infrastructure. |
| **Kubernetes** | Operators for Kafka/NATS, event-driven controllers, custom event systems. |
| **MLOps** | Training event streaming, model deployment events, pipeline trigger events. |
| **LLM Infrastructure** | Request queuing for inference, async prompt processing, event streaming for audit. |
| **Cloud Infrastructure** | Cloud event processing, infrastructure change events, audit event streaming. |
| **GPU ML Systems** | GPU event streaming, training progress events, device health events. |
| **Distributed Systems** | Message queues are foundational for distributed systems communication. |

### Practical Exercises

- [ ] Set up Kafka locally and produce/consume messages from Go
- [ ] Implement a consumer group with multiple consumers
- [ ] Handle consumer rebalancing and offset management
- [ ] Set up NATS JetStream and implement pub/sub with persistence
- [ ] Implement at-least-once delivery with idempotent consumers
- [ ] Build a dead letter queue for failed messages
- [ ] Implement the outbox pattern for reliable event publishing

### Mini Projects

- [ ] **Event-Driven Order System**: Build an event-driven system:
  - Services: Order, Payment, Inventory, Notification
  - Events: OrderCreated, PaymentProcessed, InventoryReserved, OrderShipped
  - Kafka/NATS for event streaming
  - Saga for order fulfillment
  - Dead letter queue for failures
  - Idempotent event handlers
  - Event replay capability
- [ ] **Real-Time Data Pipeline**: Build a data pipeline:
  - Producer: generates events (user actions, system metrics)
  - Kafka/NATS for transport
  - Consumer: processes, transforms, aggregates
  - Sink: writes to database and/or Elasticsearch
  - Monitoring: lag, throughput, error rate

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Design Message Queue | System Design | Hard | Core concept |
| Number of Recent Calls | LeetCode #933 | Easy | Queue/sliding window |
| Design Circular Deque | LeetCode #641 | Medium | Queue operations |
| Design Hit Counter | LeetCode #362 | Medium | Event counting |

### Interview Focused Notes

**Common Interview Questions:**
- Compare Kafka, NATS, and RabbitMQ. When would you use each?
- What are delivery guarantees? How do you achieve exactly-once?
- What is a consumer group? How does partition assignment work?
- How do you handle failed messages (dead letter queue)?
- What is the outbox pattern?
- What is event sourcing? When would you use it?
- How do you ensure idempotent message processing?

**Common Mistakes:**
- Assuming exactly-once delivery (it requires idempotent consumers)
- Not handling consumer rebalancing (message duplication during rebalance)
- Not implementing dead letter queues (losing failed messages)
- Not monitoring consumer lag (messages pile up unnoticed)
- Processing messages without idempotency (duplicate actions)

**Interviewer Expectations:**
- Knows when to use each message broker
- Understands delivery guarantees and their implications
- Implements idempotent consumers
- Designs event-driven systems with proper error handling
- Monitors queue health (lag, throughput, errors)

---

---

# STAGE 17: MLOPS INFRASTRUCTURE

---

## 17.1 ML Lifecycle and Infrastructure

### Concepts to Learn

- [ ] ML lifecycle stages: data collection → preprocessing → feature engineering → training → evaluation → deployment → monitoring → retraining
- [ ] MLOps definition: applying DevOps principles to ML systems
- [ ] ML infrastructure components:
  - Feature store: centralized feature management
  - Experiment tracking: parameters, metrics, artifacts
  - Model registry: versioned model storage with metadata
  - Training orchestration: distributed training, hyperparameter tuning
  - Model serving: real-time inference, batch inference
  - Model monitoring: data drift, model degradation, performance metrics
- [ ] Data pipeline concepts:
  - ETL (Extract, Transform, Load) vs ELT
  - Batch processing vs stream processing
  - Data versioning (DVC, LakeFS)
  - Data quality checks
- [ ] Training infrastructure:
  - Distributed training: data parallelism, model parallelism, pipeline parallelism
  - Training job scheduling: priority, preemption, gang scheduling
  - Checkpoint management: save/restore training state
  - Hyperparameter optimization: grid search, random search, Bayesian optimization

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | CI/CD for ML models, infrastructure for training and serving. |
| **Platform Engineering** | Building ML platforms for data science teams. |
| **Kubernetes** | Kubernetes is the primary orchestration platform for ML workloads. |
| **MLOps** | This IS the job. Building and operating ML infrastructure. |
| **LLM Infrastructure** | LLM training and serving is a specialized form of MLOps. |
| **Cloud Infrastructure** | Cloud-based ML infrastructure, GPU provisioning, cost optimization. |
| **GPU ML Systems** | GPU management for training and inference workloads. |
| **Distributed Systems** | Distributed training is a distributed systems problem. |

### Practical Exercises

- [ ] Set up MLflow for experiment tracking (Go client interaction via REST API)
- [ ] Build a Go service that manages model versions in a model registry
- [ ] Implement a training job scheduler in Go
- [ ] Build a data pipeline orchestrator in Go
- [ ] Create health checks for ML model serving endpoints
- [ ] Implement model A/B testing infrastructure

### Mini Projects

- [ ] **ML Model Registry**: Build a model registry service in Go:
  - REST API for CRUD model versions
  - Model metadata (framework, metrics, parameters)
  - Model artifact storage (S3/GCS integration)
  - Model stage management (staging → production → archived)
  - Model comparison (metrics diff)
  - Webhook notifications on stage transitions
  - Kubernetes CRD integration
- [ ] **Training Job Manager**: Build a training job management system:
  - Submit training jobs via API
  - Schedule on Kubernetes (create pods/jobs)
  - GPU resource management
  - Checkpoint management (save/restore)
  - Log streaming
  - Metrics collection
  - Job history and comparison

### Logic Building / DSA Problems

| Problem | Platform | Difficulty | Why |
|---------|----------|------------|-----|
| Task Scheduler | LeetCode #621 | Medium | Job scheduling |
| Process Tasks Using Servers | LeetCode #1882 | Medium | Resource-based scheduling |
| Maximum Number of Events | LeetCode #1353 | Medium | Event scheduling optimization |

### Interview Focused Notes

**Common Interview Questions:**
- What is MLOps and why is it important?
- Describe the ML lifecycle and the infrastructure needed at each stage.
- How do you manage model versions in production?
- How do you detect model drift?
- How do you implement A/B testing for ML models?
- What is a feature store?
- How do you handle distributed training on Kubernetes?

**Common Mistakes:**
- No model versioning (can't reproduce results)
- No experiment tracking (can't compare experiments)
- No monitoring in production (model degradation goes unnoticed)
- Training on development machines instead of proper infrastructure
- No automated retraining pipeline

**Interviewer Expectations:**
- Understands the full ML lifecycle
- Can design ML infrastructure components
- Knows Kubernetes-based ML tools (Kubeflow, KServe)
- Can build Go services for ML infrastructure
- Understands distributed training concepts

---

## 17.2 ML Infrastructure Tools: Kubeflow, KServe, Argo

### Concepts to Learn

**Kubeflow:**
- [ ] Kubeflow components: Pipelines, Notebooks, Training Operator, KServe, Katib
- [ ] Kubeflow Pipelines: DAG-based ML pipelines on Kubernetes
- [ ] Training Operator: TFJob, PyTorchJob, MPIJob, XGBoostJob
- [ ] Katib: hyperparameter tuning and neural architecture search
- [ ] Kubeflow CRDs and their Go types

**KServe (formerly KFServing):**
- [ ] KServe architecture: InferenceService CRD
- [ ] Serving runtimes: TensorFlow, PyTorch, Triton, custom
- [ ] Predictor, Transformer, Explainer components
- [ ] Canary deployment for models
- [ ] Auto-scaling (KPA, HPA)
- [ ] Model mesh: multi-model serving

**Argo Workflows:**
- [ ] Argo Workflow CRD: DAG and step-based workflows
- [ ] Templates: container, script, resource, DAG, steps
- [ ] Artifacts: input/output artifact management
- [ ] Parameters: workflow parameters, step parameters
- [ ] Argo Events: event-driven workflow triggers
- [ ] Argo CD: GitOps for Kubernetes (not directly ML but used in MLOps)

**Go integration with ML tools:**
- [ ] Building custom Kubeflow pipeline components in Go
- [ ] Building custom KServe transformers in Go
- [ ] Building Argo workflow templates that run Go services
- [ ] Writing Kubernetes operators that extend ML tools

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Operating Kubeflow/Argo installations, managing upgrades, monitoring. |
| **Platform Engineering** | Building ML platforms on top of Kubeflow, customizing for organization needs. |
| **Kubernetes** | These tools are all Kubernetes-native. Understanding them means understanding advanced Kubernetes patterns. |
| **MLOps** | These are the standard tools for MLOps on Kubernetes. |
| **LLM Infrastructure** | KServe for LLM serving, Argo for LLM training pipelines. |
| **Cloud Infrastructure** | Deploying and managing ML infrastructure on cloud Kubernetes. |
| **GPU ML Systems** | Training operators manage GPU jobs, KServe manages GPU inference. |
| **Distributed Systems** | These tools implement distributed system patterns on Kubernetes. |

### Practical Exercises

- [ ] Deploy Kubeflow on a kind cluster (minimal installation)
- [ ] Create a Kubeflow Pipeline that trains and deploys a model
- [ ] Deploy a model with KServe InferenceService
- [ ] Set up canary deployment for a model in KServe
- [ ] Create an Argo Workflow for a multi-step data pipeline
- [ ] Write a Go service that submits Argo Workflows via API
- [ ] Build a custom Kubeflow Training Operator job type

### Mini Projects

- [ ] **ML Platform Controller**: Build a Kubernetes operator that provides a simplified ML platform:
  - CRD: `MLPipeline` (data source, preprocessing steps, training config, serving config)
  - Creates Argo Workflows for data processing
  - Creates Training Jobs via Kubeflow Training Operator
  - Creates KServe InferenceServices for serving
  - Manages model promotion (staging → production)
  - Status tracking and notifications

### Interview Focused Notes

**Common Interview Questions:**
- What is Kubeflow and what are its components?
- How does KServe work for model serving?
- What is Argo Workflows and how is it used in MLOps?
- How do you implement canary deployment for ML models?
- How would you design an ML platform on Kubernetes?
- What are the trade-offs between Kubeflow and custom solutions?

**Common Mistakes:**
- Over-engineering ML infrastructure (using Kubeflow when simpler tools suffice)
- Not understanding the Kubernetes primitives underneath ML tools
- Not implementing proper monitoring for ML-specific metrics
- Not handling GPU resource management properly

**Interviewer Expectations:**
- Knows the ML infrastructure tool landscape
- Can deploy and configure these tools
- Can extend them with custom Go components
- Understands trade-offs between different approaches

---

---

# STAGE 18: LLMOPS INFRASTRUCTURE

---

## 18.1 LLM Serving Infrastructure

### Concepts to Learn

- [ ] LLM inference pipeline: tokenization → preprocessing → inference → post-processing → response
- [ ] LLM serving challenges:
  - Large model sizes (7B → 70B → 400B+ parameters)
  - GPU memory requirements (model weights + KV cache + activation memory)
  - Latency requirements (time to first token, tokens per second)
  - Throughput (concurrent requests, batch size)
  - Cost optimization (GPU utilization, batching efficiency)
- [ ] Serving frameworks:
  - **vLLM**: PagedAttention, continuous batching, high throughput
  - **Triton Inference Server**: multi-framework, model ensemble, dynamic batching
  - **TGI (Text Generation Inference)**: Hugging Face's serving solution
  - **TensorRT-LLM**: NVIDIA's optimized inference
  - **llama.cpp / ollama**: CPU-focused inference
- [ ] Batching strategies:
  - Static batching: fixed batch size
  - Dynamic batching: accumulate requests within a time window
  - Continuous batching (iteration-level scheduling): process requests at different stages
- [ ] KV Cache management:
  - Memory allocation for key-value cache
  - PagedAttention: virtual memory for KV cache
  - KV cache compression
- [ ] Model parallelism for serving:
  - Tensor parallelism: split layers across GPUs
  - Pipeline parallelism: split model stages across GPUs
- [ ] Quantization: FP16, INT8, INT4 (reducing model size and memory)
- [ ] Speculative decoding: draft model + verification
- [ ] Streaming responses: Server-Sent Events, WebSocket

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Deploying and operating LLM serving infrastructure. |
| **Platform Engineering** | Building LLM platforms for development teams. |
| **Kubernetes** | LLM serving on Kubernetes with GPU scheduling, auto-scaling. |
| **MLOps** | LLM is the most demanding ML workload. All MLOps applies with additional complexity. |
| **LLM Infrastructure** | This IS the job. Building and operating LLM serving systems. |
| **Cloud Infrastructure** | GPU cloud provisioning, cost optimization for LLM workloads. |
| **GPU ML Systems** | LLM serving is the primary consumer of GPU infrastructure. |
| **Distributed Systems** | Distributed LLM serving across multiple GPUs and nodes. |

### Practical Exercises

- [ ] Deploy vLLM on Kubernetes with GPU support
- [ ] Build a Go API gateway that proxies to vLLM/Triton backends
- [ ] Implement request batching in Go (accumulate requests, send as batch)
- [ ] Build a streaming proxy that converts vLLM streaming to SSE
- [ ] Implement load balancing across multiple LLM replicas with GPU-aware routing
- [ ] Build health check system for LLM serving endpoints
- [ ] Implement request queuing with priority and rate limiting

### Mini Projects

- [ ] **LLM API Gateway**: Build a production LLM API gateway in Go:
  - OpenAI-compatible API interface
  - Request routing to multiple backends (vLLM, Triton)
  - Request queuing with priority
  - Rate limiting per API key
  - Token counting and usage tracking
  - Streaming response proxy (SSE)
  - Health checking and failover
  - Metrics: latency, tokens/sec, queue depth, GPU utilization
  - Cost tracking per request
  - A/B testing between models
- [ ] **Model Deployment Controller**: Build a Kubernetes operator for LLM deployment:
  - CRD: `LLMDeployment` (model name, version, GPU requirements, replicas, quantization)
  - Creates: Deployment with GPU resources, Service, Ingress, HPA
  - Model download init container
  - Health checking (model loaded, responding)
  - Canary deployment
  - Auto-scaling based on queue depth
  - Status: model health, latency, throughput

### Interview Focused Notes

**Common Interview Questions:**
- How does LLM inference work at a high level?
- What is continuous batching and why is it important?
- What is KV cache and how is it managed?
- How do you optimize LLM serving latency?
- How do you scale LLM serving on Kubernetes?
- What is the difference between vLLM, Triton, and TGI?
- How do you handle streaming responses?

**Common Mistakes:**
- Not implementing proper batching (low GPU utilization)
- Not monitoring KV cache usage (OOM)
- Not implementing health checks (routing to unhealthy replicas)
- Not implementing request queuing (overwhelming the model server)
- Not tracking costs per request

**Interviewer Expectations:**
- Understands LLM serving architecture
- Can build infrastructure around LLM serving frameworks
- Knows GPU resource management for LLM
- Can design for high availability and performance
- Understands cost optimization strategies

---

## 18.2 Vector Databases and RAG Infrastructure

### Concepts to Learn

- [ ] Vector databases: storing and querying high-dimensional vectors
- [ ] Embedding models: converting text/images to vectors
- [ ] Similarity search: cosine similarity, euclidean distance, dot product
- [ ] Approximate Nearest Neighbor (ANN): HNSW, IVF, PQ
- [ ] Vector databases:
  - **Pinecone**: managed vector database
  - **Weaviate**: open-source, Go-based vector database
  - **Milvus**: open-source, distributed vector database
  - **Qdrant**: Rust-based, high performance
  - **pgvector**: PostgreSQL extension for vectors
  - **ChromaDB**: lightweight, Python-focused
- [ ] RAG (Retrieval-Augmented Generation):
  - Document ingestion pipeline
  - Chunking strategies (fixed size, semantic, recursive)
  - Embedding generation
  - Vector storage and indexing
  - Retrieval (similarity search + reranking)
  - Context augmentation
  - LLM generation with context
- [ ] RAG optimizations:
  - Hybrid search (vector + keyword)
  - Reranking (cross-encoder)
  - Query expansion
  - Contextual compression
  - Multi-hop retrieval
- [ ] Embedding pipeline in Go:
  - HTTP client to embedding APIs (OpenAI, Cohere, local models)
  - Batch embedding generation
  - Vector database client in Go
  - Document processing (PDF, HTML, Markdown parsing)

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Deploying vector databases, managing embedding pipelines. |
| **Platform Engineering** | RAG platform for internal teams, document search infrastructure. |
| **Kubernetes** | Vector database operators, embedding service deployment. |
| **MLOps** | Embedding pipeline management, vector database monitoring. |
| **LLM Infrastructure** | RAG is the most common LLM application pattern. This is core infrastructure. |
| **Cloud Infrastructure** | Vector database provisioning, embedding compute resources. |
| **GPU ML Systems** | GPU-accelerated embedding generation, vector indexing. |
| **Distributed Systems** | Distributed vector search, sharded embedding storage. |

### Practical Exercises

- [ ] Set up pgvector and store/query vectors from Go
- [ ] Build a Go service that generates embeddings via OpenAI API
- [ ] Implement a document chunking pipeline in Go
- [ ] Build a RAG pipeline: ingest → chunk → embed → store → retrieve → augment → generate
- [ ] Implement hybrid search (vector + full-text) with pgvector + PostgreSQL
- [ ] Benchmark different chunk sizes and overlap settings
- [ ] Implement a reranking step using a cross-encoder API

### Mini Projects

- [ ] **RAG Platform**: Build a complete RAG infrastructure in Go:
  - Document ingestion API (upload PDF, HTML, text)
  - Document processing (parsing, chunking)
  - Embedding generation (batch, async)
  - Vector storage (pgvector or Weaviate)
  - Search API (hybrid: vector + keyword)
  - RAG API (search + LLM generation)
  - Admin API (manage documents, view chunks, search analytics)
  - Metrics: search latency, relevance scores, embedding throughput
  - Kubernetes deployment with auto-scaling
- [ ] **Knowledge Base Operator**: Kubernetes operator for managing knowledge bases:
  - CRD: `KnowledgeBase` (source, chunking config, embedding model, vector store)
  - Automated document sync from sources (S3, git, web)
  - Automated re-embedding on config change
  - Health status and metrics

### Interview Focused Notes

**Common Interview Questions:**
- What is RAG and why is it used?
- How do vector databases work?
- What are embedding models and how do you choose one?
- What chunking strategies exist? How do you choose?
- How do you evaluate RAG quality?
- What is hybrid search?
- How do you handle document updates in a RAG system?

**Common Mistakes:**
- Poor chunking strategy (too large = irrelevant context, too small = lost context)
- Not implementing hybrid search (vector-only misses keyword matches)
- Not evaluating retrieval quality (bad retrieval = bad generation)
- Not handling document updates (stale embeddings)
- Embedding model mismatch (different model for indexing vs querying)

**Interviewer Expectations:**
- Understands the RAG pipeline end-to-end
- Can design and implement RAG infrastructure
- Knows vector database options and trade-offs
- Can optimize retrieval quality
- Builds production-grade embedding pipelines

---

---

# STAGE 19: GPU INFRASTRUCTURE AND ML PLATFORMS

---

## 19.1 GPU Fundamentals and Resource Management

### Concepts to Learn

- [ ] GPU architecture: streaming multiprocessors, CUDA cores, tensor cores, memory hierarchy
- [ ] CUDA basics: kernels, threads, blocks, grids, shared memory
- [ ] GPU memory types: global, shared, local, constant, texture
- [ ] GPU memory management: allocation, deallocation, memory fragmentation
- [ ] NVIDIA tools:
  - `nvidia-smi`: GPU monitoring (utilization, memory, temperature, power)
  - `nvtop`: real-time GPU monitoring
  - `nvidia-container-runtime`: GPU support in containers
  - `CUDA toolkit`: compiler, libraries, tools
- [ ] NCCL (NVIDIA Collective Communications Library): multi-GPU communication
- [ ] GPU types and capabilities:
  - Consumer: RTX series
  - Data center: A100, H100, H200, B200
  - Memory: HBM vs GDDR
  - Interconnect: NVLink, NVSwitch, InfiniBand
- [ ] Multi-GPU strategies:
  - Data parallelism: replicate model, split data
  - Model parallelism: split model across GPUs
  - Pipeline parallelism: split model stages
  - Tensor parallelism: split individual operations
- [ ] GPU in Kubernetes:
  - NVIDIA GPU Operator
  - NVIDIA Device Plugin
  - GPU resource requests: `nvidia.com/gpu: 1`
  - GPU sharing: MIG (Multi-Instance GPU), time-slicing
  - Topology-aware scheduling
- [ ] GPU monitoring: DCGM (Data Center GPU Manager), GPU metrics exporter

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Managing GPU infrastructure, monitoring GPU health, troubleshooting GPU issues. |
| **Platform Engineering** | GPU platform for ML teams, GPU sharing and quota management. |
| **Kubernetes** | GPU scheduling, device plugins, GPU operator management. |
| **MLOps** | GPU resource management for training and inference. |
| **LLM Infrastructure** | LLMs require GPUs for inference. GPU management is the core challenge. |
| **Cloud Infrastructure** | GPU cloud provisioning, GPU fleet management, cost optimization. |
| **GPU ML Systems** | This IS the domain. Every concept is directly relevant. |
| **Distributed Systems** | Multi-node GPU training, GPU cluster management. |

### Practical Exercises

- [ ] Monitor GPU usage with `nvidia-smi` and understand each metric
- [ ] Deploy NVIDIA GPU Operator on a Kubernetes cluster
- [ ] Schedule a pod with GPU requests on Kubernetes
- [ ] Set up GPU monitoring with DCGM Exporter + Prometheus + Grafana
- [ ] Build a Go service that reads GPU metrics via NVML (using CGO or REST API)
- [ ] Implement GPU-aware scheduling logic in Go
- [ ] Set up MIG on an A100 GPU and deploy multiple models

### Mini Projects

- [ ] **GPU Cluster Manager**: Build a Go service for GPU cluster management:
  - GPU inventory tracking (type, memory, utilization, health)
  - GPU allocation management (which GPU is assigned to which job)
  - GPU utilization monitoring and alerting
  - GPU sharing policy enforcement
  - Cost tracking per team/project
  - Scheduling suggestions (which GPU for which workload)
  - REST API + CLI tool
  - Prometheus metrics + Grafana dashboards
- [ ] **GPU Device Plugin**: Build a custom Kubernetes device plugin:
  - Advertise custom GPU resources
  - Implement Allocate RPC
  - Health checking for GPU devices
  - Topology hints for NUMA-aware scheduling
  - Metrics reporting

### Interview Focused Notes

**Common Interview Questions:**
- How does GPU scheduling work in Kubernetes?
- What is the NVIDIA GPU Operator?
- What is MIG and how does it help with GPU sharing?
- How do you monitor GPU health and utilization?
- What is the difference between data parallelism and model parallelism?
- How do you handle GPU memory fragmentation?
- What is NVLink and why does it matter?

**Common Mistakes:**
- Not monitoring GPU utilization (expensive GPUs sitting idle)
- Not implementing GPU health checks (training on faulty GPU)
- Not using GPU sharing (each job gets a whole GPU even if it doesn't need it)
- Not considering GPU topology in scheduling (NVLink, PCIe)
- Not implementing proper cleanup on job failure (GPU memory leak)

**Interviewer Expectations:**
- Understands GPU architecture at a practical level
- Can manage GPU resources in Kubernetes
- Implements monitoring and health checking
- Knows GPU sharing strategies
- Can design GPU scheduling policies

---

## 19.2 Triton Inference Server and Serving Frameworks

### Concepts to Learn

- [ ] Triton Inference Server architecture:
  - Model repository: model storage format
  - Backend support: TensorRT, ONNX, PyTorch, TensorFlow, Python, custom
  - Model configuration: `config.pbtxt`
  - Dynamic batching: configurable batch strategy
  - Model ensemble: chaining models
  - Model management: load/unload models at runtime
  - Metrics: built-in Prometheus metrics
  - Health endpoints: readiness, liveness
- [ ] Triton client in Go:
  - HTTP/REST API client
  - gRPC API client (`triton-inference-server/client`)
  - Inference request/response handling
  - Streaming inference
- [ ] Ray Serve:
  - Actor-based serving
  - Composition graphs
  - Auto-scaling
  - Integration with Ray clusters
- [ ] Model optimization:
  - ONNX conversion
  - TensorRT optimization
  - Quantization (FP16, INT8, INT4)
  - Model distillation
  - Pruning

### Why This Matters in Real Jobs

| Domain | Relevance |
|--------|-----------|
| **DevOps** | Deploying and operating Triton/serving infrastructure. |
| **Platform Engineering** | Model serving platform, multi-model serving, model management. |
| **Kubernetes** | Triton on Kubernetes with GPU scheduling, auto-scaling. |
| **MLOps** | Model serving is the final stage of MLOps. |
| **LLM Infrastructure** | Triton is widely used for LLM serving (with TensorRT-LLM backend). |
| **Cloud Infrastructure** | Serving infrastructure provisioning, GPU allocation for inference. |
| **GPU ML Systems** | Triton is the primary GPU inference server. |
| **Distributed Systems** | Distributed inference, model ensemble routing. |

### Practical Exercises

- [ ] Deploy Triton Inference Server on Kubernetes
- [ ] Build a Go client that sends inference requests to Triton (HTTP and gRPC)
- [ ] Configure dynamic batching in Triton
- [ ] Set up model ensemble in Triton
- [ ] Build a Go service that manages model loading/unloading
- [ ] Implement A/B testing between model versions
- [ ] Monitor Triton with Prometheus and Grafana

### Mini Projects

- [ ] **Model Serving Platform**: Build a complete model serving platform:
  - Go API for model management (upload, deploy, undeploy, list)
  - Triton as the serving backend
  - Kubernetes operator for InferenceService lifecycle
  - A/B testing and canary deployment
  - Auto-scaling based on request rate and latency
  - Multi-model serving (shared GPU via Triton)
  - Monitoring dashboard (latency, throughput, GPU utilization)
  - Model performance comparison

### Interview Focused Notes

**Common Interview Questions:**
- What is Triton Inference Server and why is it used?
- How does dynamic batching work?
- How do you optimize model serving performance?
- What is model ensemble in Triton?
- How do you handle model updates in production?
- What is quantization and how does it affect serving?

**Common Mistakes:**
- Not using dynamic batching (low GPU utilization)
- Not monitoring model performance in production
- Not implementing proper health checks
- Not handling model loading failures gracefully
- Not considering GPU memory when loading multiple models

**Interviewer Expectations:**
- Can deploy and configure Triton
- Builds Go clients for inference
- Understands optimization techniques
- Designs for high availability and performance
- Monitors model serving quality

---

---

# STAGE 20: SYSTEM DESIGN FOR INFRASTRUCTURE INTERVIEWS

---

## 20.1 System Design Methodology

### Concepts to Learn

- [ ] System design interview framework:
  1. **Clarify requirements** (functional, non-functional, scale)
  2. **Estimate scale** (users, requests/sec, data size, bandwidth)
  3. **Define API** (endpoints, request/response)
  4. **Design high-level architecture** (components, data flow)
  5. **Deep dive into components** (database, cache, queue, etc.)
  6. **Address scalability** (horizontal scaling, sharding, caching)
  7. **Address reliability** (replication, failover, monitoring)
  8. **Address security** (auth, encryption, rate limiting)

- [ ] Back-of-the-envelope estimation:
  - Powers of 2 (KB, MB, GB, TB, PB)
  - Latency numbers: L1 cache (1ns), L2 (4ns), RAM (100ns), SSD (100us), HDD (10ms), network same datacenter (0.5ms), cross-region (100ms)
  - QPS estimates, storage estimates, bandwidth estimates

### Why This Matters in Real Jobs

Every senior+ infrastructure role will have system design interviews. These problems directly mirror real production challenges.

### System Design Problems

#### Infrastructure-Focused:

- [ ] **Design a Kubernetes Platform**
  - Multi-tenant cluster with namespace isolation
  - GitOps deployment pipeline
  - Admission controllers for policy enforcement
  - Observability stack (logging, metrics, tracing)
  - Service mesh for inter-service communication
  - Secrets management
  - Autoscaling (cluster and workload)
  - Disaster recovery

- [ ] **Design an ML Training Platform**
  - Training job submission API
  - GPU scheduling with priority and preemption
  - Distributed training support
  - Experiment tracking and model registry
  - Checkpoint management
  - Cost tracking and optimization
  - Auto-scaling GPU nodes

- [ ] **Design an LLM Inference Platform**
  - Multi-model serving with GPU management
  - Request routing and load balancing
  - Auto-scaling based on queue depth
  - KV cache management
  - Streaming response delivery
  - A/B testing between models
  - Cost tracking per API key
  - Rate limiting and billing

- [ ] **Design a Distributed Key-Value Store**
  - Raft consensus for replication
  - Consistent hashing for sharding
  - Read/write path
  - Snapshot and recovery
  - Compaction
  - Linearizable reads

- [ ] **Design a CI/CD Pipeline System**
  - Pipeline definition (YAML)
  - Stage execution (containers)
  - Artifact management
  - Parallel and sequential stages
  - Secret injection
  - Caching
  - Multi-environment deployment

- [ ] **Design a Monitoring and Alerting System**
  - Metric collection (push/pull)
  - Time-series database
  - Alert rules engine
  - Notification routing
  - Dashboards
  - SLO tracking

- [ ] **Design a Container Orchestration System** (simplified Kubernetes)
  - Node management
  - Scheduling
  - Service discovery
  - Health checking
  - Rolling updates
  - Resource management

- [ ] **Design a Service Mesh**
  - Sidecar proxy
  - Traffic management
  - mTLS
  - Observability
  - Load balancing
  - Circuit breaking

- [ ] **Design a GPU Cluster Management System**
  - GPU inventory tracking
  - Job scheduling with GPU affinity
  - GPU sharing (MIG, time-slicing)
  - Health monitoring
  - Cost allocation
  - Auto-scaling

- [ ] **Design a Real-Time Feature Store**
  - Online and offline stores
  - Feature ingestion pipeline
  - Point-in-time lookups
  - Feature freshness SLA
  - Feature versioning

### Interview Focused Notes

**For each system design problem, address:**
1. Functional requirements
2. Non-functional requirements (latency, throughput, availability, consistency)
3. Scale estimates
4. API design
5. Data model
6. High-level architecture diagram
7. Deep dive into 2-3 critical components
8. Trade-offs and alternatives considered
9. Monitoring and alerting
10. Failure scenarios and mitigation

**Common Mistakes:**
- Jumping to solution without clarifying requirements
- Not estimating scale
- Over-engineering for scale that isn't needed
- Ignoring failure scenarios
- Not discussing trade-offs
- Not mentioning monitoring and observability

**Interviewer Expectations:**
- Structured approach to design
- Clear communication of trade-offs
- Realistic scale estimation
- Depth in areas of expertise
- Awareness of operational concerns (monitoring, debugging, deployment)

---

---

# STAGE 21: CAPSTONE PROJECTS

---

## 21.1 Capstone Project 1: Kubernetes Application Operator

### Architecture
```
┌─────────────────────────────────────────────────────┐
│                 Kubernetes Cluster                    │
│                                                      │
│  ┌─────────────────┐    ┌─────────────────┐         │
│  │   Application    │    │   Application    │         │
│  │   Controller     │    │   Webhook        │         │
│  │                  │    │  (Validate/      │         │
│  │  Reconcile Loop  │    │   Default)       │         │
│  └────────┬─────────┘    └─────────────────┘         │
│           │                                           │
│           ▼                                           │
│  ┌─────────────────────────────────────────┐         │
│  │  Creates/Manages:                        │         │
│  │  - Deployment   - Service                │         │
│  │  - Ingress      - HPA                    │         │
│  │  - PDB          - ConfigMap              │         │
│  │  - Secret       - NetworkPolicy          │         │
│  └─────────────────────────────────────────┘         │
│                                                      │
│  ┌─────────────────┐    ┌─────────────────┐         │
│  │  Prometheus      │    │  Grafana         │         │
│  │  Metrics         │    │  Dashboard       │         │
│  └─────────────────┘    └─────────────────┘         │
└─────────────────────────────────────────────────────┘
```

### Tech Stack
- Go, Kubebuilder, controller-runtime
- CRD with OpenAPI validation
- Admission webhooks (validating + mutating)
- Prometheus metrics, structured logging
- envtest for integration tests

### Requirements
- [ ] CRD: `Application` with spec for image, replicas, ports, env, resources, ingress, autoscaling
- [ ] Controller: creates Deployment, Service, Ingress, HPA, PDB, NetworkPolicy
- [ ] Validating webhook: ensures valid configuration
- [ ] Mutating webhook: applies defaults
- [ ] Status: conditions, URL, replicas, health
- [ ] Canary deployment support
- [ ] Rollback support
- [ ] Metrics: reconciliation latency, error rate, resource counts
- [ ] Complete test suite: unit + envtest integration
- [ ] Helm chart for operator deployment
- [ ] Documentation with examples

### Scaling Considerations
- Leader election for HA
- Cache tuning for large clusters
- Rate-limited reconciliation
- Efficient status updates

### Failure Handling
- Partial resource creation recovery
- Finalizer for cleanup
- Status degraded conditions
- Error classification (retryable vs permanent)

### Security Considerations
- Minimal RBAC permissions
- Webhook TLS
- Input validation on all CRD fields
- No privilege escalation

### Interview Talking Points
- "I built a Kubernetes operator using Kubebuilder that automates application deployment with full lifecycle management"
- "Implements the reconciliation pattern with idempotent operations and proper error handling"
- "Includes canary deployments, auto-scaling, and network policies"
- "Full test coverage with envtest integration tests"

---

## 21.2 Capstone Project 2: DevOps Automation Platform

### Architecture
```
┌─────────────────────────────────────────────────────┐
│                    DevOps Platform                    │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │  CLI      │  │  Web UI   │  │  GitHub Webhook  │  │
│  │  (Cobra)  │  │  (API)    │  │  Receiver        │  │
│  └─────┬─────┘  └─────┬─────┘  └────────┬─────────┘  │
│        │              │                  │            │
│        ▼              ▼                  ▼            │
│  ┌─────────────────────────────────────────┐         │
│  │            API Gateway (Gin)             │         │
│  │  Auth | Rate Limit | Logging | Tracing  │         │
│  └──────────────────┬──────────────────────┘         │
│                     │                                 │
│        ┌────────────┼────────────┐                   │
│        ▼            ▼            ▼                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ Deploy   │ │ Pipeline │ │ Config   │            │
│  │ Service  │ │ Service  │ │ Service  │            │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘            │
│       │            │            │                    │
│       ▼            ▼            ▼                    │
│  ┌────────┐  ┌──────────┐  ┌──────────┐            │
│  │  K8s   │  │  Argo    │  │  Vault   │            │
│  │  API   │  │Workflows │  │          │            │
│  └────────┘  └──────────┘  └──────────┘            │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │PostgreSQL│  │  Redis   │  │  Kafka   │          │
│  └──────────┘  └──────────┘  └──────────┘          │
└─────────────────────────────────────────────────────┘
```

### Tech Stack
- Go, Gin, Cobra/Viper, pgx, go-redis, kafka-go
- Kubernetes client-go, Argo Workflows API
- Vault for secrets, PostgreSQL for state, Redis for cache
- Prometheus + OTel for observability

### Requirements
- [ ] CLI tool for deployment management (deploy, rollback, status, logs)
- [ ] REST API for all operations
- [ ] GitHub webhook integration (auto-deploy on merge)
- [ ] Pipeline execution (build, test, deploy stages)
- [ ] Multi-environment support (dev, staging, production)
- [ ] Secrets management via Vault
- [ ] Audit logging for all operations
- [ ] Deployment history and comparison
- [ ] Rollback with automatic health check
- [ ] Notification system (Slack, email)
- [ ] Metrics and tracing

### Scaling Considerations
- Horizontal scaling of API servers
- Kafka for async pipeline execution
- Redis for caching deployment status
- PostgreSQL read replicas for analytics

### Failure Handling
- Deployment rollback on health check failure
- Pipeline retry with exponential backoff
- Circuit breaker for external services
- Dead letter queue for failed notifications

### Security Considerations
- JWT authentication for API
- RBAC for operations (who can deploy to production)
- Secrets never logged or exposed
- Audit trail for compliance

### Interview Talking Points
- "Built a complete DevOps platform with CLI, API, and webhook-driven automation"
- "Implements GitOps with automatic deployment on merge to main"
- "Supports multi-environment deployment with rollback and health checking"
- "Clean architecture with DI, full test coverage, observability"

---

## 21.3 Capstone Project 3: ML Training Platform

### Architecture
```
┌─────────────────────────────────────────────────────┐
│                 ML Training Platform                  │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │  CLI      │  │  Web API  │  │  Scheduler       │  │
│  └─────┬─────┘  └─────┬─────┘  │  (GPU-aware)     │  │
│        │              │         └────────┬─────────┘  │
│        ▼              ▼                  │            │
│  ┌─────────────────────────────────────────┐         │
│  │          Training Job Manager            │         │
│  │  Submit | Monitor | Cancel | Retry       │         │
│  └──────────────────┬──────────────────────┘         │
│                     │                                 │
│        ┌────────────┼────────────┐                   │
│        ▼            ▼            ▼                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │Experiment│ │  Model   │ │Checkpoint│            │
│  │ Tracker  │ │ Registry │ │ Manager  │            │
│  └──────────┘ └──────────┘ └──────────┘            │
│                                                      │
│  ┌──────────────────────────────────────────┐       │
│  │           Kubernetes Cluster              │       │
│  │  ┌──────────┐  ┌──────────┐              │       │
│  │  │  GPU Node │  │  GPU Node │              │       │
│  │  │  A100 x8  │  │  H100 x8  │              │       │
│  │  └──────────┘  └──────────┘              │       │
│  └──────────────────────────────────────────┘       │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │PostgreSQL│  │ S3/MinIO │  │Prometheus │          │
│  └──────────┘  └──────────┘  └──────────┘          │
└─────────────────────────────────────────────────────┘
```

### Tech Stack
- Go, Kubernetes client-go, controller-runtime
- PostgreSQL (pgx), S3 (MinIO), Prometheus
- CRDs for TrainingJob, Experiment, Model

### Requirements
- [ ] Training job submission API (model type, dataset, hyperparameters, GPU requirements)
- [ ] GPU-aware scheduling (type, count, topology)
- [ ] Distributed training support (multi-node, multi-GPU)
- [ ] Experiment tracking (parameters, metrics, artifacts)
- [ ] Model registry (version, stage, metadata)
- [ ] Checkpoint management (auto-save, resume, S3 storage)
- [ ] Log streaming from training jobs
- [ ] Cost tracking per experiment/team
- [ ] Priority and preemption for job scheduling
- [ ] Metrics: GPU utilization, training progress, cost
- [ ] CLI tool for job management

### Scaling Considerations
- Gang scheduling for distributed training
- Checkpoint-based preemption and resume
- Node auto-scaling for GPU nodes
- Efficient checkpoint storage and retrieval

### Failure Handling
- Automatic retry on transient failures
- Checkpoint resume on node failure
- GPU health check before scheduling
- Alert on training divergence (loss spike)

### Security Considerations
- Team-based access control
- Data isolation between teams
- Secure model artifact storage
- Audit logging for compliance

### Interview Talking Points
- "Built an ML training platform with GPU-aware scheduling and distributed training support"
- "Implements checkpoint management for fault-tolerant training"
- "Kubernetes operator for training job lifecycle management"
- "Full experiment tracking with cost attribution per team"

---

## 21.4 Capstone Project 4: LLM Inference Platform

### Architecture
```
┌─────────────────────────────────────────────────────┐
│               LLM Inference Platform                  │
│                                                      │
│  ┌──────────────────────────────────────────┐       │
│  │       API Gateway (OpenAI Compatible)     │       │
│  │  Auth | Rate Limit | Usage Track | Route  │       │
│  └──────────────────┬───────────────────────┘       │
│                     │                                 │
│        ┌────────────┼────────────┐                   │
│        ▼            ▼            ▼                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │  vLLM    │ │  vLLM    │ │  Triton  │            │
│  │ (Llama)  │ │ (Mistral)│ │ (Custom) │            │
│  │ GPU: 4xA100│ GPU: 2xA100│ GPU: 1xH100│          │
│  └──────────┘ └──────────┘ └──────────┘            │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │  Model   │  │  Vector  │  │  Prompt  │          │
│  │  Manager │  │  Store   │  │  Cache   │          │
│  └──────────┘  └──────────┘  └──────────┘          │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │PostgreSQL│  │  Redis   │  │Prometheus │          │
│  └──────────┘  └──────────┘  └──────────┘          │
└─────────────────────────────────────────────────────┘
```

### Tech Stack
- Go, Gin, gRPC, pgx, go-redis
- vLLM/Triton backends
- Kubernetes operator for model deployment
- Prometheus + Grafana for monitoring

### Requirements
- [ ] OpenAI-compatible REST API (completions, chat, embeddings)
- [ ] Multi-model routing (different models on different GPUs)
- [ ] Streaming response (SSE)
- [ ] Request queuing with priority
- [ ] Rate limiting per API key
- [ ] Token counting and usage billing
- [ ] Model A/B testing
- [ ] RAG integration (vector store + retrieval)
- [ ] Prompt caching (semantic similarity-based)
- [ ] Auto-scaling based on queue depth
- [ ] Model management (deploy, undeploy, update)
- [ ] Kubernetes operator for LLMDeployment CRD
- [ ] Comprehensive monitoring (latency, tokens/sec, GPU utilization, cost)

### Scaling Considerations
- Horizontal scaling of API gateway
- Auto-scaling model replicas based on demand
- GPU utilization optimization (continuous batching via vLLM)
- Cache hit rate optimization

### Failure Handling
- Model health checking and automatic failover
- Request retry on transient failures
- Circuit breaker for model backends
- Graceful degradation (fallback to smaller model)

### Security Considerations
- API key authentication
- Rate limiting and abuse prevention
- Prompt injection detection (basic)
- Data privacy (no logging of prompts/responses by default)
- PII detection

### Interview Talking Points
- "Built a production LLM inference platform with OpenAI-compatible API"
- "Supports multi-model routing, streaming, and auto-scaling"
- "Kubernetes operator for automated model deployment and lifecycle management"
- "Comprehensive observability with cost tracking per customer"

---

## 21.5 Capstone Project 5: Distributed Observability System

### Architecture
```
┌─────────────────────────────────────────────────────┐
│            Distributed Observability System           │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │  Log     │  │  Metric  │  │  Trace   │          │
│  │Collector │  │Collector │  │Collector │          │
│  │(Agent)   │  │(Agent)   │  │(Agent)   │          │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘          │
│        │             │              │                │
│        ▼             ▼              ▼                │
│  ┌──────────────────────────────────────────┐       │
│  │         OTel Collector (Pipeline)         │       │
│  │   Receive → Process → Export              │       │
│  └──────────────────┬───────────────────────┘       │
│                     │                                │
│        ┌────────────┼────────────┐                  │
│        ▼            ▼            ▼                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │  Log     │ │  Metric  │ │  Trace   │           │
│  │  Store   │ │  Store   │ │  Store   │           │
│  │ (Custom) │ │(Prom/VictM)│ │ (Jaeger) │           │
│  └──────────┘ └──────────┘ └──────────┘           │
│                                                     │
│  ┌──────────────────────────────────────────┐      │
│  │           Query & Alert Engine            │      │
│  │  PromQL | LogQL | TraceQL                │      │
│  └──────────────────┬───────────────────────┘      │
│                     │                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ Dashboard│  │  Alert   │  │  SLO     │         │
│  │  Engine  │  │  Manager │  │  Tracker │         │
│  └──────────┘  └──────────┘  └──────────┘         │
└─────────────────────────────────────────────────────┘
```

### Tech Stack
- Go for all components
- gRPC for inter-component communication
- PostgreSQL/ClickHouse for log storage
- VictoriaMetrics/Prometheus for metric storage
- Jaeger for trace storage

### Requirements
- [ ] Log collection agent (file tailing, container logs)
- [ ] Metric collection (pull-based Prometheus-compatible)
- [ ] Trace collection (OTLP receiver)
- [ ] Processing pipeline (filter, transform, aggregate, sample)
- [ ] Storage backends (pluggable)
- [ ] Query API (logs, metrics, traces)
- [ ] Alert rules engine (threshold, anomaly)
- [ ] Notification routing (Slack, email, PagerDuty)
- [ ] SLO tracking and error budget
- [ ] Dashboard API
- [ ] Correlation: link logs ↔ metrics ↔ traces via trace ID

### Interview Talking Points
- "Built a complete observability platform from scratch in Go"
- "Implements all three pillars: logs, metrics, traces with correlation"
- "Includes alert engine, SLO tracking, and notification routing"
- "Distributed architecture with pluggable storage backends"

---

## 21.6 Capstone Project 6: GPU Cluster Management Tool

### Architecture
```
┌─────────────────────────────────────────────────────┐
│            GPU Cluster Management Tool                │
│                                                      │
│  ┌──────────┐  ┌──────────┐                         │
│  │  CLI      │  │  Web API  │                         │
│  └─────┬─────┘  └─────┬─────┘                         │
│        │              │                               │
│        ▼              ▼                               │
│  ┌─────────────────────────────────────────┐         │
│  │        GPU Management Service            │         │
│  │  Inventory | Schedule | Monitor | Report │         │
│  └──────────────────┬──────────────────────┘         │
│                     │                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ Device   │  │ Scheduler│  │ Monitor  │          │
│  │ Manager  │  │(GPU-aware)│  │ (DCGM)  │          │
│  └──────────┘  └──────────┘  └──────────┘          │
│                                                      │
│  ┌──────────────────────────────────────────┐       │
│  │           GPU Cluster Nodes               │       │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  │       │
│  │  │ Node 1  │  │ Node 2  │  │ Node 3  │  │       │
│  │  │ 8xA100  │  │ 8xH100  │  │ 4xA100  │  │       │
│  │  └─────────┘  └─────────┘  └─────────┘  │       │
│  └──────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────┘
```

### Tech Stack
- Go, Cobra/Viper (CLI), Gin (API)
- Kubernetes client-go, NVML (via REST or CGO)
- PostgreSQL, Prometheus, Grafana

### Requirements
- [ ] GPU inventory management (auto-discovery, type, memory, health)
- [ ] GPU-aware job scheduling (type affinity, topology, gang scheduling)
- [ ] GPU utilization monitoring (real-time metrics via DCGM)
- [ ] GPU sharing policies (MIG configuration, time-slicing)
- [ ] Cost tracking and allocation per team
- [ ] Capacity planning and forecasting
- [ ] GPU health alerting (temperature, ECC errors, utilization)
- [ ] Maintenance mode (drain GPU node, reschedule jobs)
- [ ] CLI tool for GPU cluster management
- [ ] Grafana dashboards for GPU fleet

### Interview Talking Points
- "Built a GPU cluster management platform with intelligent scheduling"
- "Implements GPU-aware scheduling with topology awareness and gang scheduling"
- "Real-time monitoring with DCGM integration and alerting"
- "Cost tracking and capacity planning for GPU fleet"

---

---

# APPENDIX: PROGRESS TRACKING CHECKLIST

## Stage Completion

- [ ] Stage 1: Core Foundations
- [ ] Stage 2: Go Language Fundamentals
- [ ] Stage 3: Go Idioms and Best Practices
- [ ] Stage 4: Object-Oriented Design in Go
- [ ] Stage 5: Advanced Go (Concurrency and Runtime)
- [ ] Stage 6: Go for Systems Programming
- [ ] Stage 7: Go Networking Deep Dive
- [ ] Stage 8: API Development
- [ ] Stage 9: Databases and Data Persistence
- [ ] Stage 10: Testing, Benchmarking, and Profiling
- [ ] Stage 11: Go for DevOps
- [ ] Stage 12: Security in Go
- [ ] Stage 13: Containers and Kubernetes Fundamentals
- [ ] Stage 14: Kubernetes Controllers and Operators
- [ ] Stage 15: Observability and Instrumentation
- [ ] Stage 16: Distributed Systems
- [ ] Stage 17: MLOps Infrastructure
- [ ] Stage 18: LLMOps Infrastructure
- [ ] Stage 19: GPU Infrastructure
- [ ] Stage 20: System Design
- [ ] Stage 21: Capstone Projects

## Capstone Completion

- [ ] Capstone 1: Kubernetes Application Operator
- [ ] Capstone 2: DevOps Automation Platform
- [ ] Capstone 3: ML Training Platform
- [ ] Capstone 4: LLM Inference Platform
- [ ] Capstone 5: Distributed Observability System
- [ ] Capstone 6: GPU Cluster Management Tool

## DSA Progress

| Stage | Easy | Medium | Hard | Total |
|-------|------|--------|------|-------|
| 1-3   | 30+  | 15+    | 3+   | 48+   |
| 4-7   | 10+  | 20+    | 5+   | 35+   |
| 8-11  | 10+  | 15+    | 3+   | 28+   |
| 12-16 | 5+   | 10+    | 5+   | 20+   |
| 17-19 | 3+   | 5+     | 0    | 8+    |
| **Total** | **58+** | **65+** | **16+** | **139+** |

## Key Libraries to Master

| Library | Stage | Purpose |
|---------|-------|---------|
| `net/http` | 6, 8 | HTTP server/client |
| `encoding/json` | 2 | JSON serialization |
| `context` | 5 | Cancellation, timeout |
| `sync` | 5 | Concurrency primitives |
| `testing` | 10 | Testing framework |
| Cobra + Viper | 11 | CLI tools |
| Gin / Chi | 8 | Web framework |
| pgx / sqlx | 9 | Database |
| go-redis | 9 | Redis client |
| zap / slog | 15 | Structured logging |
| Prometheus client | 15 | Metrics |
| OpenTelemetry | 15 | Distributed tracing |
| client-go | 13 | Kubernetes client |
| controller-runtime | 14 | Kubernetes controllers |
| gRPC-Go | 7 | RPC framework |
| testify | 10 | Test assertions |
| golang-jwt | 12 | JWT auth |
| kafka-go / sarama | 16 | Message queue |

---

**This roadmap will take you from absolute Go beginner to a staff-level infrastructure engineer capable of building production Kubernetes platforms, distributed systems, and ML/LLM infrastructure.**

*Last updated: 2026-03-06*
