# AI Cloud Infrastructure Engineer Roadmap

> A production-grade, interview-ready roadmap covering Foundations through Expert-level mastery for AI Cloud Infrastructure Engineering roles.

---

## Section 1 — Foundations

### 1.1 Operating System Basics
- **What an OS does**: process management, memory management, file systems, I/O, security
- **Kernel vs Userspace**: syscall boundary, privilege rings (ring 0 vs ring 3)
- **Boot process**: BIOS/UEFI → bootloader → kernel → init/systemd
- **Process lifecycle**: fork → exec → wait → exit
- **Threads vs Processes**: shared memory vs isolated address spaces, pthreads, green threads

### 1.2 Networking Basics
- **OSI Model**: L1–L7, understand what happens at each layer
- **TCP/IP Model**: Link → Internet → Transport → Application
- **IP addressing**: IPv4, IPv6, subnetting, CIDR notation
- **DNS resolution flow**: stub resolver → recursive → root → TLD → authoritative
- **HTTP basics**: methods, status codes, headers, request/response lifecycle

### 1.3 Version Control (Git)
- **Core concepts**: commits, branches, merges, rebases, HEAD, refs
- **Branching strategies**: trunk-based, GitFlow, GitHub Flow
- **Merge vs Rebase**: tradeoffs, when to use each
- **Conflict resolution**: manual merge, 3-way merge algorithm
- **Git internals**: objects (blob, tree, commit, tag), packfiles, reflog

### 1.4 Programming Fundamentals
- **Languages to know**: Python (scripting/automation), Go (tooling/operators), Bash (glue)
- **Data structures**: arrays, hashmaps, trees, queues, stacks
- **Algorithms**: sorting, searching, graph traversal (BFS/DFS)
- **Concurrency**: threads, goroutines, async/await, race conditions, locks, mutexes
- **Error handling**: exceptions, error codes, retry patterns

### 1.5 YAML / JSON / TOML
- **YAML**: anchors, aliases, multiline strings, gotchas (Norway problem)
- **JSON**: schema validation, JSON Patch, JSON Path
- **TOML**: config file standard, comparison with YAML

### 1.6 Shell & CLI Proficiency
- **Bash scripting**: variables, loops, conditionals, functions, exit codes
- **Text processing**: grep, awk, sed, cut, sort, uniq, xargs, jq
- **Process management**: ps, top, htop, kill, nohup, &, jobs
- **File permissions**: chmod, chown, umask, SUID/SGID, sticky bit
- **Package managers**: apt, yum, brew, pip, go modules

### 1.7 Interview Questions — Foundations
- What happens when you type `google.com` in a browser?
- Explain the difference between a process and a thread.
- How does DNS resolution work end to end?
- What is a syscall? Give examples.
- Explain TCP 3-way handshake.
- What is the difference between `fork()` and `exec()`?
- How does Git store data internally?

---

## Section 2 — Linux

### 2.1 Kernel Internals

#### Level 0 — Basics
- **Kernel types**: monolithic (Linux) vs microkernel (Mach, QNX)
- **Kernel modules**: `lsmod`, `modprobe`, `insmod`, `rmmod`
- **Syscall interface**: how userspace talks to kernel, `strace` for tracing
- **Kernel parameters**: `/proc/sys/`, `sysctl`, persistent via `/etc/sysctl.conf`

#### Level 1 — Intermediate
- **Kernel compilation**: custom kernel builds, `.config`, `make menuconfig`
- **Kernel ring buffer**: `dmesg`, log levels, boot messages
- **eBPF**: extended Berkeley Packet Filter, in-kernel programmability
- **Kernel tracing**: ftrace, perf, bpftrace

#### Level 2 — Advanced
- **Scheduler internals**: CFS (Completely Fair Scheduler), runqueues, vruntime
- **Memory management**: page tables, TLB, page faults, OOM killer, NUMA
- **I/O subsystem**: block layer, I/O schedulers (mq-deadline, BFQ, kyber)
- **VFS layer**: virtual filesystem, inodes, dentries, superblock

### 2.2 Process Management
- **Process states**: Running, Sleeping (D/S), Stopped, Zombie
- **Process tree**: PID 1 (init/systemd), parent-child relationships, orphans
- **Signals**: SIGTERM, SIGKILL, SIGHUP, SIGUSR1/2, signal handlers
- **Priority/Niceness**: nice values (-20 to 19), `renice`, real-time priorities
- **Process limits**: `ulimit`, `/etc/security/limits.conf`, cgroup limits
- **Tools**: `ps aux`, `pstree`, `top`, `htop`, `/proc/<pid>/`

### 2.3 Memory Management
- **Virtual memory**: address spaces, page tables, MMU translation
- **Memory zones**: DMA, Normal, HighMem
- **Swap**: swap partition vs swapfile, swappiness tuning, when swap is bad
- **OOM Killer**: `oom_score`, `oom_score_adj`, how to protect critical processes
- **Huge pages**: transparent huge pages (THP), explicit huge pages, tradeoffs
- **NUMA**: Non-Uniform Memory Access, `numactl`, node-local allocation
- **Memory debugging**: `free -m`, `/proc/meminfo`, `vmstat`, `smem`, `valgrind`

### 2.4 Filesystem
- **Filesystem types**: ext4, XFS, Btrfs, tmpfs, overlayfs (Docker)
- **Inode structure**: metadata, data blocks, hard links vs soft links
- **Journaling**: write-ahead log, crash recovery, journal modes (ordered, writeback, journal)
- **Mount options**: noatime, nodiratime, discard (TRIM), barrier
- **Disk management**: `fdisk`, `parted`, `lvm`, RAID levels (0,1,5,6,10)
- **LVM**: physical volumes, volume groups, logical volumes, snapshots, thin provisioning
- **File I/O**: buffered vs direct I/O, mmap, page cache, `sync`, `fsync`

### 2.5 Linux Networking
- **Network stack**: socket → TCP/UDP → IP → device driver → NIC
- **Configuration**: `ip addr`, `ip route`, `ip link`, `ss`, `netstat`
- **iptables/nftables**: chains (INPUT, OUTPUT, FORWARD), tables (filter, nat, mangle)
- **Network namespaces**: `ip netns`, veth pairs, bridges
- **Bonding/Teaming**: active-backup, LACP, load balancing modes
- **tc (traffic control)**: queueing disciplines, rate limiting, traffic shaping
- **Tools**: `tcpdump`, `wireshark`, `nmap`, `traceroute`, `mtr`, `dig`, `curl`

### 2.6 Cgroups and Namespaces (Container Foundations)

#### Cgroups (Control Groups)
- **v1 vs v2**: hierarchy differences, unified hierarchy in v2
- **Controllers**: cpu, memory, io, pids, cpuset
- **CPU limits**: `cpu.cfs_quota_us`, `cpu.cfs_period_us`, cpu shares
- **Memory limits**: `memory.limit_in_bytes`, `memory.memsw.limit_in_bytes`
- **I/O limits**: blkio controller, IOPS/bandwidth throttling
- **Management**: `cgcreate`, `cgset`, `cgexec`, systemd slices

#### Namespaces
- **Types**: PID, NET, MNT, UTS, IPC, USER, CGROUP, TIME
- **PID namespace**: isolated PID trees, PID 1 per namespace
- **NET namespace**: separate network stacks, veth pairs for connectivity
- **MNT namespace**: isolated mount points, pivot_root
- **USER namespace**: UID/GID mapping, rootless containers
- **Tools**: `unshare`, `nsenter`, `lsns`

### 2.7 Performance Tuning
- **CPU**: governor settings, IRQ affinity, CPU pinning, isolcpus
- **Memory**: vm.swappiness, vm.dirty_ratio, vm.overcommit_memory, THP tuning
- **Disk**: I/O scheduler selection, readahead, filesystem mount options
- **Network**: `net.core.somaxconn`, `net.ipv4.tcp_tw_reuse`, buffer sizes, conntrack table size
- **Tools**: `perf`, `flamegraphs`, `sar`, `iostat`, `mpstat`, `pidstat`
- **USE method**: Utilization, Saturation, Errors for each resource

### 2.8 Systemd
- **Unit types**: service, timer, socket, mount, target, slice
- **Service management**: `systemctl start/stop/restart/status/enable/disable`
- **Unit files**: `[Unit]`, `[Service]`, `[Install]` sections
- **Dependencies**: Requires, Wants, After, Before
- **Resource control**: CPUQuota, MemoryMax, IOWeight (cgroup integration)
- **Journald**: `journalctl`, log filtering, persistent storage

### 2.9 Security
- **SELinux**: enforcing/permissive/disabled, contexts, policies, `audit2allow`
- **AppArmor**: profiles, complain/enforce modes
- **Capabilities**: `CAP_NET_ADMIN`, `CAP_SYS_ADMIN`, dropping privileges
- **Seccomp**: syscall filtering, BPF profiles
- **PAM**: pluggable authentication modules
- **Audit**: `auditd`, audit rules, log analysis
- **Hardening**: CIS benchmarks, SSH hardening, kernel hardening (KASLR, SMEP, SMAP)

### 2.10 Failure Scenarios & Debugging
- **High CPU**: identify with `top`/`perf`, check for CPU steal (virtualization), runaway processes
- **Memory leak**: growing RSS, OOM kills in `dmesg`, use `smem`, `pmap`
- **Disk full**: `df -h`, `du -sh`, find large files, check inode exhaustion with `df -i`
- **Zombie processes**: parent not calling `wait()`, fix parent or reparent
- **Network unreachable**: check routing (`ip route`), DNS (`dig`), firewall (`iptables -L`), MTU
- **High load average**: CPU-bound vs I/O-bound (check `wa` in `top`), D-state processes
- **Kernel panic**: analyze crash dump, check `dmesg`, hardware issues

### 2.11 Interview Questions — Linux
- How does the OOM killer decide which process to kill?
- Explain how cgroups v2 differs from v1.
- What are namespaces and how do containers use them?
- A server has load average of 100 but CPU usage is low — what's happening?
- How would you debug a process in D (uninterruptible sleep) state?
- Explain the Linux boot process from power-on to login prompt.
- What happens when you run `ls -la` at the kernel level?
- How does the CFS scheduler work?
- Explain the difference between hard and soft limits in ulimit.
- How would you tune a Linux server for a high-throughput network application?

### 2.12 Hands-on Labs — Linux
- [ ] Build a custom kernel and boot into it
- [ ] Create a container from scratch using namespaces and cgroups
- [ ] Write eBPF program to trace syscalls
- [ ] Set up LVM with thin provisioning and snapshots
- [ ] Configure iptables rules for a multi-tier application
- [ ] Tune kernel parameters for 10K+ connections
- [ ] Debug a simulated memory leak using /proc and smem
- [ ] Set up network namespaces with veth pairs and bridge

---

## Section 3 — Networking

### 3.1 TCP/IP Deep Dive

#### Level 0 — Basics
- **TCP/IP model layers**: Link (L2) → Internet (L3) → Transport (L4) → Application (L7)
- **IP basics**: IPv4 (32-bit), IPv6 (128-bit), packet structure (header + payload), TTL, fragmentation
- **Ports**: well-known (0–1023), registered (1024–49151), ephemeral (49152–65535)
- **Sockets**: combination of IP + port + protocol, `SOCK_STREAM` (TCP) vs `SOCK_DGRAM` (UDP)
- **Encapsulation**: data → segment (TCP header) → packet (IP header) → frame (Ethernet header + trailer)

#### Level 1 — Intermediate
- **3-way handshake (connection establishment)**:
  - Client sends `SYN` (seq=x)
  - Server responds `SYN-ACK` (seq=y, ack=x+1)
  - Client sends `ACK` (ack=y+1)
  - Connection is now `ESTABLISHED` on both sides
  - SYN queue (half-open connections) and accept queue (fully established, waiting for `accept()`)
  - `SYN_SENT`, `SYN_RECEIVED` intermediate states
- **4-way teardown (connection termination)**:
  - Initiator sends `FIN` → enters `FIN_WAIT_1`
  - Receiver sends `ACK` → enters `CLOSE_WAIT`; initiator enters `FIN_WAIT_2`
  - Receiver sends `FIN` → enters `LAST_ACK`
  - Initiator sends `ACK` → enters `TIME_WAIT` (waits 2*MSL before closing)
  - `TIME_WAIT` purpose: ensure delayed packets do not corrupt new connections on the same tuple
  - Half-close: one side can close its send direction while still receiving
- **TCP states (full lifecycle)**:
  - `LISTEN` → `SYN_SENT` → `SYN_RECEIVED` → `ESTABLISHED`
  - `FIN_WAIT_1` → `FIN_WAIT_2` → `TIME_WAIT` → `CLOSED`
  - `CLOSE_WAIT` → `LAST_ACK` → `CLOSED`
  - Use `ss -tan` or `netstat -tan` to observe states
  - `CLOSE_WAIT` accumulation indicates the application is not calling `close()` — a common bug
- **Flow control**:
  - Receiver advertises a **receive window** (`rwnd`) in every ACK
  - Sender must not have more unacknowledged bytes in flight than `rwnd`
  - Window size of 0 triggers **zero-window probes** (persist timer) to avoid deadlock
  - **Window scaling** (RFC 7323): TCP option negotiated in SYN/SYN-ACK, allows window sizes up to 1 GB (scale factor 0–14, shifts the 16-bit window field)
  - Without window scaling, max window is 65,535 bytes — insufficient for high-bandwidth-delay-product links

#### Level 2 — Advanced
- **Congestion control**:
  - **Slow start**: `cwnd` starts at 1 MSS (or IW=10 in modern stacks), doubles each RTT until `ssthresh`
  - **Congestion avoidance**: linear growth (additive increase) after `ssthresh`
  - **Fast retransmit**: 3 duplicate ACKs trigger immediate retransmit without waiting for timeout
  - **Fast recovery**: after fast retransmit, halve `cwnd` and enter congestion avoidance (skip slow start)
  - **Algorithms**:
    - **Reno**: classic, halve cwnd on loss
    - **CUBIC** (Linux default): uses cubic function for window growth, optimized for high-BDP networks
    - **BBR** (Google): model-based, estimates bottleneck bandwidth and RTT, avoids filling buffers
    - BBR vs CUBIC: BBR achieves higher throughput on lossy links, CUBIC is loss-based
  - Tuning: `net.ipv4.tcp_congestion_control`, `net.core.default_qdisc`
- **Nagle's algorithm**:
  - Buffers small outgoing segments and coalesces them to reduce overhead from small packets
  - Sends immediately only if: (a) segment is full-sized (MSS), or (b) all prior data has been ACKed
  - Can cause latency for interactive protocols (SSH, real-time APIs)
  - Disable with `TCP_NODELAY` socket option
  - Interaction with **delayed ACKs** can cause 40ms+ latency spikes (Nagle waits for ACK, receiver delays ACK)
- **Delayed ACKs**:
  - Receiver delays ACK up to ~40ms hoping to piggyback ACK on a data response
  - Reduces ACK traffic but increases latency when combined with Nagle's algorithm
  - Disable with `TCP_QUICKACK` socket option
  - `net.ipv4.tcp_delack_min` tuning (non-standard, some kernels)
- **Selective Acknowledgment (SACK)**:
  - Allows receiver to inform sender about non-contiguous blocks received
  - Avoids retransmitting already-received segments
  - Negotiated via TCP options in SYN; `net.ipv4.tcp_sack`
- **TCP timestamps (RFC 7323)**:
  - Used for RTTM (Round-Trip Time Measurement) and PAWS (Protection Against Wrapped Sequence numbers)
  - Negotiated in SYN/SYN-ACK
  - `net.ipv4.tcp_timestamps`
- **MSS (Maximum Segment Size)**:
  - Negotiated during handshake, typically 1460 bytes for Ethernet (1500 MTU - 20 IP - 20 TCP)
  - Path MTU Discovery (PMTUD): uses ICMP "Fragmentation Needed" to find path MTU
  - Black hole issue: firewalls dropping ICMP can break PMTUD
- **TCP keepalive**:
  - Detects dead connections via periodic probes
  - `net.ipv4.tcp_keepalive_time` (default 7200s), `tcp_keepalive_intvl`, `tcp_keepalive_probes`
  - Application-level keepalives (HTTP/2 PING, gRPC keepalive) are often preferred

### 3.2 UDP

#### Level 0 — Basics
- **Characteristics**: connectionless, no handshake, no guaranteed delivery, no ordering, no congestion control
- **Header**: 8 bytes — source port, destination port, length, checksum
- **Use cases**: when low latency matters more than reliability — DNS, VoIP, video streaming, gaming, metrics collection
- **Comparison with TCP**: no retransmissions, no flow/congestion control, lower overhead, higher throughput for loss-tolerant workloads

#### Level 1 — Intermediate
- **DNS over UDP**: most DNS queries use UDP on port 53, fall back to TCP when response exceeds 512 bytes (or 4096 with EDNS0)
- **DHCP**: uses UDP (ports 67/68), broadcast-based
- **SNMP**: monitoring protocol over UDP (port 161/162)
- **Syslog**: traditionally UDP 514, modern setups use TCP for reliability
- **UDP and NAT**: conntrack tracks UDP "connections" by tuple with timeout (default 30s for unreplied, 120s for assured)
- **UDP buffer tuning**: `net.core.rmem_max`, `net.core.wmem_max`, application `SO_RCVBUF`

#### Level 2 — Advanced
- **QUIC protocol**:
  - Built on UDP, provides TLS 1.3, multiplexed streams, 0-RTT connection establishment
  - Eliminates head-of-line blocking at the transport layer (unlike HTTP/2 over TCP)
  - Built-in connection migration (survives IP changes, e.g., mobile Wi-Fi to cellular)
  - Used by HTTP/3
  - Congestion control implemented in userspace (pluggable: CUBIC, BBR)
  - Encrypted transport headers — middleboxes cannot inspect or modify
- **RTP/RTCP**: Real-time Transport Protocol for audio/video, RTCP provides feedback (jitter, loss)
- **UDP amplification attacks**: DNS amplification, NTP monlist, memcached — reflection attacks using spoofed source IPs
- **Mitigations**: BCP 38 ingress filtering, rate limiting, response rate limiting (DNS RRL)

### 3.3 DNS

#### Level 0 — Basics
- **What DNS does**: translates domain names to IP addresses (and reverse)
- **Resolution flow (iterative)**:
  1. Application calls `getaddrinfo()` / stub resolver
  2. Stub resolver checks `/etc/hosts`, then local cache
  3. Query goes to configured recursive resolver (`/etc/resolv.conf`)
  4. Recursive resolver queries root nameservers (`.`)
  5. Root responds with TLD nameserver referral (e.g., `.com`)
  6. TLD responds with authoritative nameserver referral (e.g., `ns1.example.com`)
  7. Authoritative nameserver returns the answer
  8. Recursive resolver caches and returns to client
- **Record types**:
  - `A`: maps name to IPv4 address
  - `AAAA`: maps name to IPv6 address
  - `CNAME`: canonical name alias (cannot coexist with other records at same name, except DNSSEC)
  - `MX`: mail exchange servers with priority
  - `TXT`: arbitrary text, used for SPF, DKIM, DMARC, domain verification
  - `SRV`: service location records (priority, weight, port, target) — used by Kubernetes, SIP, LDAP
  - `NS`: delegates a zone to nameservers
  - `PTR`: reverse DNS lookup (IP to name)
  - `SOA`: Start of Authority — zone metadata (serial, refresh, retry, expire, minimum TTL)

#### Level 1 — Intermediate
- **TTL (Time To Live)**:
  - How long resolvers/clients cache a record (in seconds)
  - Low TTL (30–60s): faster failover, higher query load on authoritative servers
  - High TTL (3600–86400s): reduced query load, slower failover
  - Best practice: lower TTL before DNS changes, raise after propagation
  - Negative caching: TTL for NXDOMAIN responses (SOA minimum field)
- **DNS caching layers**:
  - Browser cache (Chrome: `chrome://net-internals/#dns`)
  - OS resolver cache (`systemd-resolved`, `nscd`, `dnsmasq`)
  - Recursive resolver cache (ISP, corporate, 8.8.8.8, 1.1.1.1)
  - CDN/edge resolver cache
- **Zone file structure**: SOA, NS records, A/AAAA/CNAME records, `$ORIGIN`, `$TTL`
- **Zone transfer**: AXFR (full), IXFR (incremental), restricted by ACLs
- **EDNS0 (Extension mechanisms for DNS)**: extends UDP payload size beyond 512 bytes (typically 4096)
- **Glue records**: A records for nameservers within the zone they serve (avoids circular dependency)
- **Wildcard records**: `*.example.com` — matches any subdomain not explicitly defined

#### Level 2 — Advanced
- **DNS-over-HTTPS (DoH)**:
  - DNS queries over HTTPS (port 443), encrypted and authenticated
  - Prevents ISP/network-level DNS snooping
  - Bypasses traditional DNS-based filtering — controversial in enterprise environments
  - Supported by Firefox, Chrome, Cloudflare (1.1.1.1/dns-query), Google (dns.google/resolve)
- **DNS-over-TLS (DoT)**:
  - DNS queries encrypted with TLS on port 853
  - Easier to block/detect than DoH (dedicated port)
  - Supported by `systemd-resolved`, `stubby`, Android 9+
- **Split-horizon DNS (split-brain DNS)**:
  - Returns different answers based on source IP / network location
  - Internal clients get private IPs, external clients get public IPs
  - Common in corporate environments and cloud VPCs
  - Implementation: separate views in BIND, Route 53 private hosted zones, CoreDNS plugins
- **DNSSEC**:
  - Provides authentication and integrity for DNS responses (not confidentiality)
  - Chain of trust: root → TLD → authoritative (DS and DNSIG records)
  - `RRSIG`, `DNSKEY`, `DS`, `NSEC`/`NSEC3` record types
- **DNS load balancing**: round-robin A records, weighted routing, latency-based routing (Route 53), GeoDNS
- **Service discovery via DNS**: Kubernetes uses CoreDNS — `svc.cluster.local`, headless services return pod IPs, `SRV` records for port discovery
- **Common DNS issues**:
  - Stale cache causing traffic to old IPs
  - CNAME at zone apex breaking standards
  - TTL too high during migration causing split traffic
  - DNS amplification attacks from open resolvers

### 3.4 HTTP/HTTPS

#### Level 0 — Basics
- **HTTP methods**: GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS
- **Status codes**: 1xx (informational), 2xx (success), 3xx (redirect), 4xx (client error), 5xx (server error)
- **Common codes**: 200, 201, 204, 301, 302, 304, 400, 401, 403, 404, 429, 500, 502, 503, 504
- **Headers**: `Content-Type`, `Authorization`, `Cache-Control`, `User-Agent`, `Accept`, `Host`
- **Request/response model**: client opens connection, sends request, server sends response
- **URLs**: scheme://host:port/path?query#fragment

#### Level 1 — Intermediate
- **HTTP/1.1**:
  - **Persistent connections (keep-alive)**: reuse TCP connection for multiple requests
  - **Pipelining**: send multiple requests without waiting for responses (rarely used in practice due to head-of-line blocking)
  - **Chunked transfer encoding**: streaming responses with `Transfer-Encoding: chunked`
  - **Head-of-line blocking**: responses must be returned in order, a slow response blocks all subsequent ones
  - **Connection limits**: browsers limit to 6 concurrent connections per origin → domain sharding workaround
- **HTTP/2**:
  - **Binary framing layer**: replaces textual HTTP/1.1 with binary frames
  - **Multiplexing**: multiple streams on a single TCP connection, eliminates HTTP-level head-of-line blocking
  - **Stream prioritization**: weight-based prioritization of concurrent streams
  - **Header compression (HPACK)**: static and dynamic table, Huffman encoding, eliminates redundant headers
  - **Server push**: server proactively sends resources before client requests them (deprecated in most browsers)
  - **Single TCP connection per origin**: reduces connection overhead, but TCP-level head-of-line blocking remains
- **HSTS (HTTP Strict Transport Security)**:
  - `Strict-Transport-Security` header: forces HTTPS for all future requests
  - `max-age`: duration (seconds) to remember the policy
  - `includeSubDomains`: applies to all subdomains
  - HSTS preload list: hardcoded in browsers, submit via `hstspreload.org`
  - Prevents SSL stripping attacks

#### Level 2 — Advanced
- **HTTP/3**:
  - Built on QUIC (UDP-based transport)
  - Eliminates TCP head-of-line blocking — stream loss only affects that stream
  - 0-RTT connection establishment (when resuming)
  - Built-in TLS 1.3 (no separate TLS handshake)
  - Connection migration: survives IP address changes
  - Alt-Svc header: server advertises HTTP/3 support, client upgrades
- **HTTP caching**:
  - `Cache-Control`: `max-age`, `s-maxage`, `no-cache`, `no-store`, `public`, `private`, `must-revalidate`
  - `ETag` / `If-None-Match`: conditional requests, returns 304 if unchanged
  - `Last-Modified` / `If-Modified-Since`: timestamp-based conditional requests
  - Vary header: cache key includes specified request headers (e.g., `Vary: Accept-Encoding`)
  - Stale-while-revalidate: serve stale content while revalidating in background
- **Proxy protocols**:
  - **Forward proxy**: client-side, used for filtering, caching, anonymity
  - **Reverse proxy**: server-side, used for load balancing, SSL termination, caching (nginx, HAProxy, Envoy)
  - **PROXY protocol**: preserves original client IP through L4 proxies (HAProxy, AWS NLB)
- **WebSocket**: full-duplex communication over single TCP connection, upgrade from HTTP/1.1, used for real-time applications
- **gRPC**: HTTP/2-based RPC framework, Protobuf serialization, bidirectional streaming, deadline propagation

### 3.5 TLS/SSL

#### Level 0 — Basics
- **Purpose**: confidentiality (encryption), integrity (MAC), authentication (certificates)
- **TLS versions**: TLS 1.0/1.1 (deprecated), TLS 1.2 (widely used), TLS 1.3 (current, faster, more secure)
- **Certificate structure**: subject, issuer, validity period, public key, signature, serial number, SANs
- **PKI (Public Key Infrastructure)**: CA (Certificate Authority) → intermediate CA → leaf certificate
- **Certificate types**: DV (Domain Validation), OV (Organization Validation), EV (Extended Validation)

#### Level 1 — Intermediate
- **TLS 1.2 handshake flow**:
  1. `ClientHello`: supported cipher suites, TLS version, random, SNI extension
  2. `ServerHello`: chosen cipher suite, random
  3. Server sends certificate chain
  4. `ServerKeyExchange` (for DHE/ECDHE): server's DH public value
  5. `ServerHelloDone`
  6. `ClientKeyExchange`: client's DH public value (or RSA-encrypted pre-master secret)
  7. Both derive master secret → session keys
  8. `ChangeCipherSpec` + `Finished` (both sides)
  9. Application data flows encrypted
  - **2-RTT handshake** (1-RTT with session resumption via session tickets)
- **TLS 1.3 handshake flow**:
  1. `ClientHello`: supported groups, key shares, signature algorithms
  2. `ServerHello`: chosen key share
  3. Server sends `EncryptedExtensions`, certificate, `CertificateVerify`, `Finished`
  4. Client sends `Finished`
  - **1-RTT handshake** (0-RTT with PSK resumption — replay risk)
  - Removed: RSA key exchange, static DH, custom DHE groups, compression, renegotiation
- **Certificate chain verification**:
  - Leaf cert → intermediate CA(s) → root CA (in trust store)
  - Each cert's signature is verified against the issuer's public key
  - Check: validity period, revocation status, hostname match (CN or SAN)
  - Missing intermediate cert is the most common TLS misconfiguration
- **SNI (Server Name Indication)**:
  - TLS extension: client includes the hostname in `ClientHello`
  - Allows multiple HTTPS sites on a single IP (virtual hosting for TLS)
  - Sent in plaintext in TLS 1.2 — Encrypted SNI (ESNI/ECH) in TLS 1.3 draft
- **ALPN (Application-Layer Protocol Negotiation)**:
  - TLS extension: negotiates application protocol during handshake (e.g., `h2`, `http/1.1`)
  - Required for HTTP/2 — browser only uses HTTP/2 over TLS with ALPN

#### Level 2 — Advanced
- **Cipher suites**:
  - Format (TLS 1.2): `TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384`
    - Key exchange: ECDHE
    - Authentication: RSA
    - Encryption: AES-256-GCM
    - MAC: SHA384
  - TLS 1.3 simplified: `TLS_AES_256_GCM_SHA384` (key exchange always ECDHE, auth via signature)
  - **Forward secrecy**: ECDHE/DHE generates ephemeral keys per session — compromised server key does not decrypt past traffic
  - Disable weak ciphers: RC4, DES, 3DES, MD5, SHA1, export ciphers
- **Mutual TLS (mTLS)**:
  - Both client and server present certificates
  - Server sends `CertificateRequest` during handshake
  - Used in: service mesh (Istio, Linkerd), zero-trust architectures, API authentication
  - Certificate distribution: SPIFFE/SPIRE, Vault PKI, cert-manager
- **Certificate pinning**:
  - Client validates server cert against a pinned public key or certificate hash
  - Prevents MITM even if a CA is compromised
  - HTTP Public Key Pinning (HPKP) — deprecated due to risk of self-DoS
  - Still used in mobile apps and internal services
- **OCSP (Online Certificate Status Protocol)**:
  - Real-time certificate revocation checking
  - **OCSP stapling**: server fetches OCSP response and includes it in TLS handshake — avoids client-side OCSP lookup latency and privacy leak
  - **CRL (Certificate Revocation List)**: downloaded list of revoked certs — large, less timely
  - **CRLite**: compressed revocation data (Firefox)
- **Testing TLS**:
  - `openssl s_client -connect host:443 -servername host` — inspect handshake, certificate chain
  - `openssl x509 -in cert.pem -text -noout` — decode certificate details
  - SSL Labs (ssllabs.com) — comprehensive TLS assessment
  - `testssl.sh` — CLI tool for TLS testing

### 3.6 Load Balancing

#### Level 0 — Basics
- **Purpose**: distribute traffic across multiple backends for availability, scalability, fault tolerance
- **L4 (Transport layer) load balancing**:
  - Operates on TCP/UDP connection level
  - Routes based on IP + port, does not inspect HTTP content
  - Lower latency, higher throughput — no application parsing overhead
  - Examples: AWS NLB, HAProxy (TCP mode), IPVS, LVS
- **L7 (Application layer) load balancing**:
  - Inspects HTTP headers, URL paths, cookies, gRPC methods
  - Enables content-based routing, URL rewriting, header manipulation
  - Terminates TLS, can add/modify headers (X-Forwarded-For, X-Request-ID)
  - Examples: AWS ALB, nginx, HAProxy (HTTP mode), Envoy, Traefik

#### Level 1 — Intermediate
- **Algorithms**:
  - **Round-robin**: requests distributed sequentially; simple, even spread for homogeneous backends
  - **Weighted round-robin**: backends get traffic proportional to assigned weight
  - **Least connections**: route to backend with fewest active connections; best for variable request duration
  - **Weighted least connections**: combines least-connections with weights
  - **IP hash**: hash client IP to select backend; provides basic session affinity
  - **Consistent hashing**: hash ring — adding/removing nodes minimizes remapping; used in caches, distributed systems
  - **Random with two choices**: pick 2 random backends, choose the one with fewer connections — surprisingly effective
- **Health checks**:
  - **Active**: LB periodically probes backends (TCP connect, HTTP GET, gRPC health)
  - **Passive**: LB detects failures from real traffic (error rates, timeouts)
  - Health check parameters: interval, timeout, healthy/unhealthy thresholds
  - Separate health check endpoints: `/healthz` (liveness), `/readyz` (readiness)
  - Graceful degradation: mark unhealthy but keep in rotation if all backends fail
- **Session affinity (sticky sessions)**:
  - Cookie-based: LB inserts cookie identifying backend
  - Source IP-based: hash client IP (breaks with NAT/proxies)
  - Header-based: route on custom header value
  - Tradeoff: enables stateful apps but causes uneven load distribution
  - Best practice: design stateless services, externalize session state (Redis, database)

#### Level 2 — Advanced
- **Connection draining (deregistration delay)**:
  - When removing a backend, allow in-flight requests to complete
  - New connections are not sent to draining backend
  - Configurable timeout (e.g., AWS ALB default 300s)
  - Critical for zero-downtime deployments and rolling updates
- **SSL/TLS termination**:
  - Terminate TLS at the load balancer — backend communication in plaintext or re-encrypted
  - SSL offloading reduces backend CPU load
  - SSL passthrough: LB forwards encrypted traffic to backend (L4) — backend handles TLS
  - Re-encryption: LB terminates client TLS, initiates new TLS to backend (defense in depth)
- **Global server load balancing (GSLB)**:
  - DNS-based: return IPs of the closest/healthiest region
  - Anycast: same IP announced from multiple locations, BGP routes to nearest
  - Examples: AWS Route 53, Cloudflare, Google Cloud Load Balancing
- **Direct Server Return (DSR)**:
  - Response bypasses the load balancer, sent directly from backend to client
  - LB only handles inbound traffic — significant throughput improvement for asymmetric traffic
  - Backend must have the VIP configured on a loopback interface
- **Load balancer failure modes**:
  - Single LB is a SPOF — use HA pair (active-passive with VRRP/keepalived) or managed LB
  - Thundering herd on LB restart — connection storms
  - Health check false positives — avoid overly aggressive checks

### 3.7 VPC Networking

#### Level 0 — Basics
- **VPC (Virtual Private Cloud)**: isolated virtual network within a cloud provider
- **CIDR block**: defines the IP address range for the VPC (e.g., `10.0.0.0/16` = 65,536 IPs)
- **Subnets**: subdivisions of VPC CIDR, each in a specific availability zone
  - **Public subnet**: has a route to an internet gateway
  - **Private subnet**: no direct route to internet, uses NAT gateway for outbound
- **Route tables**: rules that determine where network traffic is directed
  - Each subnet is associated with exactly one route table
  - Local route (VPC CIDR) is implicit and cannot be removed
- **Internet Gateway (IGW)**: VPC component allowing communication between VPC and the internet
  - Attached to VPC, referenced in route table (`0.0.0.0/0 → igw-xxx`)
  - Performs 1:1 NAT for instances with public IPs

#### Level 1 — Intermediate
- **NAT Gateway**:
  - Managed NAT service for private subnets to reach the internet (outbound only)
  - Placed in a public subnet, private subnet route table points `0.0.0.0/0 → nat-gw-xxx`
  - Supports up to 55,000 simultaneous connections per destination
  - NAT gateway per AZ for high availability
  - Cost: per-hour charge + per-GB data processing charge
- **VPC Peering**:
  - Direct network connection between two VPCs (same or different accounts/regions)
  - Non-transitive: VPC-A peered with VPC-B and VPC-B peered with VPC-C does not mean VPC-A can reach VPC-C
  - No overlapping CIDR blocks allowed
  - Requires route table entries in both VPCs
  - Cross-region peering: traffic encrypted, but higher latency
- **VPC Endpoints**:
  - **Gateway endpoints**: for S3 and DynamoDB, free, route table entry
  - **Interface endpoints (PrivateLink)**: ENI in your subnet with private IP, for other AWS services and custom services
  - Keeps traffic within the AWS network — no internet traversal
  - DNS resolution: endpoint-specific DNS or private DNS overrides service default
- **Security layers**:
  - **Security groups**: instance-level, stateful firewall (return traffic automatically allowed)
  - **NACLs (Network ACLs)**: subnet-level, stateless firewall (must explicitly allow return traffic)
  - Security groups: allow rules only; NACLs: allow and deny rules
  - Evaluation order: NACLs first (numbered rules, lowest first), then security groups

#### Level 2 — Advanced
- **Transit Gateway**:
  - Hub-and-spoke network hub connecting multiple VPCs, VPNs, and Direct Connect
  - Transitive routing: VPC-A → Transit GW → VPC-B (unlike peering)
  - Route tables on transit gateway for segmentation
  - Supports inter-region peering
  - Scales to thousands of VPCs
  - Replaces complex mesh peering topologies
- **PrivateLink (endpoint services)**:
  - Expose a service in your VPC to other VPCs/accounts without peering
  - Service provider creates NLB + VPC endpoint service
  - Consumer creates interface endpoint — gets a private IP in their VPC
  - Traffic never traverses the public internet
  - One-directional: consumer initiates connections to provider
  - Used by SaaS providers, internal platform services
- **VPC Flow Logs**:
  - Capture metadata about IP traffic: source/dest IP, ports, protocol, action (ACCEPT/REJECT), bytes
  - Levels: VPC, subnet, or ENI
  - Destinations: CloudWatch Logs, S3, Kinesis Data Firehose
  - Does not capture packet payloads — use traffic mirroring for that
  - Use for: security analysis, troubleshooting connectivity, compliance
- **Elastic Network Interfaces (ENI)**:
  - Virtual network card attached to an instance
  - Has: private IP(s), optional public IP, MAC address, security groups
  - Can be detached and re-attached to another instance (failover pattern)
  - Multiple ENIs for multi-homed instances (management + data plane separation)
- **VPC design patterns**:
  - Multi-AZ for high availability
  - Separate VPCs for prod/staging/dev (account-per-environment preferred)
  - Shared services VPC (DNS, logging, monitoring) connected via transit gateway
  - Hub-and-spoke with inspection VPC (firewall appliance) for centralized traffic inspection

### 3.8 Routing

#### Level 0 — Basics
- **Static routing**: manually configured routes, no protocol overhead, suitable for simple topologies
  - `ip route add 10.0.0.0/8 via 192.168.1.1`
  - Default route: `0.0.0.0/0 via <gateway>`
- **Dynamic routing**: routes learned automatically via routing protocols, adapts to topology changes
- **Routing table**: ordered list of destination prefixes and next hops
- **Longest prefix match**: more specific route wins (e.g., `/32` beats `/24` beats `/16` beats `/0`)
- **Administrative distance**: preference among routing sources (connected=0, static=1, OSPF=110, BGP=20 eBGP/200 iBGP)

#### Level 1 — Intermediate
- **BGP (Border Gateway Protocol)**:
  - Path vector protocol, the routing protocol of the internet
  - **eBGP**: between different Autonomous Systems (AS), default TTL=1
  - **iBGP**: within the same AS, full mesh or route reflectors
  - Uses TCP port 179
  - Route selection: longest prefix → highest local preference → shortest AS path → lowest MED → eBGP over iBGP → lowest router ID
  - **BGP communities**: tags for route policy (no-export, no-advertise, custom)
  - **BGP peering**: requires explicit neighbor configuration, establishes TCP session
  - **AS-PATH prepending**: makes a route look longer to influence inbound traffic
  - Cloud relevance: AWS Direct Connect, GCP Cloud Router, Azure ExpressRoute all use BGP
- **OSPF (Open Shortest Path First)**:
  - Link-state protocol, uses Dijkstra's algorithm for shortest path
  - **Areas**: area 0 (backbone) + other areas, reduces routing table size and LSA flooding
  - **Router types**: internal, ABR (area border), ASBR (AS boundary)
  - **LSA types**: Router (Type 1), Network (Type 2), Summary (Type 3), External (Type 5)
  - Hello packets for neighbor discovery and keepalive (default 10s interval, 40s dead)
  - Cost metric based on interface bandwidth
  - Used in large on-prem data center networks

#### Level 2 — Advanced
- **Route tables in cloud**:
  - AWS: VPC route tables, transit gateway route tables, subnet associations
  - Route propagation: BGP routes from VPN/Direct Connect automatically added
  - Most specific route wins — blackhole routes for preventing traffic leaks
  - Policy-based routing: route based on source IP, not just destination (limited in cloud, use SDN/overlay)
- **Equal-cost multi-path (ECMP)**:
  - Multiple next hops for the same prefix — traffic distributed across paths
  - Per-flow hashing (5-tuple) to avoid packet reordering
  - Common in data center leaf-spine topologies
  - Transit gateway supports ECMP across VPN tunnels
- **Route leaking**:
  - Selectively sharing routes between routing domains (e.g., VRF, transit gateway route tables)
  - Use case: allow shared services to be reachable from isolated environments
- **Overlay networking**:
  - VXLAN: L2 over L3, 24-bit VNI allowing 16M segments, UDP encapsulation (port 4789)
  - GENEVE: successor to VXLAN, extensible with TLV options, used by AWS (Nitro)
  - IPIP, GRE: simpler tunneling protocols
  - Kubernetes: pod networks (Calico, Cilium, Flannel) use overlay or direct routing

### 3.9 NAT (Network Address Translation)

#### Level 0 — Basics
- **Purpose**: translates private IPs to public IPs and vice versa; conserves IPv4 address space
- **SNAT (Source NAT)**: rewrites source IP — used for outbound traffic from private networks
- **DNAT (Destination NAT)**: rewrites destination IP — used for inbound traffic (port forwarding, load balancing)
- **PAT (Port Address Translation / NAT overload / masquerade)**:
  - Many internal IPs share one external IP, differentiated by source port
  - The most common NAT type (home routers, cloud NAT gateways)

#### Level 1 — Intermediate
- **Masquerade**:
  - Special case of SNAT where the source IP is dynamically determined from the outbound interface
  - Used when the external IP can change (DHCP, PPPoE)
  - `iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE`
  - Slightly slower than SNAT (looks up interface IP per packet)
- **Conntrack (connection tracking)**:
  - Kernel module (`nf_conntrack`) that tracks NAT state and stateful firewall rules
  - Tracks connection state: `NEW`, `ESTABLISHED`, `RELATED`, `INVALID`
  - Conntrack table: hash table of tracked connections
  - `conntrack -L` — list tracked connections
  - `net.netfilter.nf_conntrack_max` — maximum tracked connections (default often 65536)
  - Conntrack table exhaustion causes packet drops — monitor `nf_conntrack_count` vs `nf_conntrack_max`
  - Per-protocol timeouts: TCP established (432000s), UDP (30s unreplied, 120s assured)
- **NAT in iptables**:
  - `PREROUTING` chain: DNAT (before routing decision)
  - `POSTROUTING` chain: SNAT/MASQUERADE (after routing decision)
  - `OUTPUT` chain: DNAT for locally generated traffic
  - Example DNAT: `iptables -t nat -A PREROUTING -p tcp --dport 80 -j DNAT --to-destination 10.0.0.5:8080`
  - Example SNAT: `iptables -t nat -A POSTROUTING -s 10.0.0.0/24 -o eth0 -j SNAT --to-source 203.0.113.1`

#### Level 2 — Advanced
- **NAT in cloud environments**:
  - AWS NAT Gateway: managed SNAT for private subnets, 55K simultaneous connections per destination per NAT IP
  - Port exhaustion: use multiple NAT gateways or multiple Elastic IPs per NAT gateway
  - GCP Cloud NAT: per-VM port allocation, configurable min ports per VM
  - Azure NAT Gateway: supports multiple public IPs, 64K SNAT ports per IP
- **NAT and Kubernetes**:
  - Pod-to-external: SNAT at node level (iptables MASQUERADE) — external services see node IP
  - `externalTrafficPolicy: Local`: preserves client source IP, avoids extra SNAT hop, but uneven load
  - Service ClusterIP: DNAT via iptables/IPVS to pod IPs
  - NodePort: DNAT from node external IP:port to pod IP:port
- **NAT traversal**:
  - NAT breaks end-to-end connectivity — inbound connections to NATed hosts require DNAT/port forwarding
  - STUN: discover external IP and port mapping
  - TURN: relay traffic when direct NAT traversal fails
  - ICE: tries direct, STUN, then TURN — used by WebRTC
- **Hairpin NAT (NAT loopback)**:
  - Internal host accessing a public IP that DNATs back into the same network
  - Requires both DNAT and SNAT to avoid asymmetric routing
  - Common issue in self-hosted environments

### 3.10 VPN

#### Level 0 — Basics
- **Purpose**: encrypted tunnel over untrusted networks, extends private networks securely
- **Types**:
  - **Site-to-site**: connects two networks (data center to cloud, VPC to VPC)
  - **Client VPN (remote access)**: individual device connects to a network
- **Tunnel vs transport mode**:
  - **Tunnel mode**: entire IP packet encrypted and encapsulated (new outer IP header) — site-to-site
  - **Transport mode**: only payload encrypted, original IP header preserved — host-to-host

#### Level 1 — Intermediate
- **IPSec**:
  - **IKE Phase 1**: authenticates peers, establishes IKE SA (ISAKMP SA)
    - Main mode (6 messages, identity protected) or Aggressive mode (3 messages, faster, identity exposed)
    - Authentication: pre-shared key (PSK), certificates, or EAP
  - **IKE Phase 2**: negotiates IPSec SAs for data encryption
    - Quick mode: establishes two unidirectional SAs (one per direction)
  - **IKEv2**: simplified (4 messages for initial exchange), supports MOBIKE (mobility), more reliable
  - **ESP (Encapsulating Security Payload)**: provides encryption + integrity + anti-replay
  - **AH (Authentication Header)**: integrity + authentication only (no encryption, rarely used)
  - **SA (Security Association)**: unidirectional, identified by SPI (Security Parameter Index)
  - **PFS (Perfect Forward Secrecy)**: new DH exchange per Phase 2 — compromise of one session key does not affect others
- **WireGuard**:
  - Modern, minimal VPN protocol (~4,000 lines of code vs 100K+ for OpenVPN/IPSec)
  - Uses Curve25519 (DH), ChaCha20-Poly1305 (encryption), BLAKE2s (hashing)
  - UDP-based, single port
  - Cryptokey routing: allowed IPs mapped to public keys
  - Kernel module (Linux) — very high performance
  - No certificate infrastructure needed — just public/private key pairs
  - Stateless: no connection state, automatic handshake and roaming
  - Cloud adoption: Tailscale, Netmaker, AWS Client VPN (OpenVPN-based, but WireGuard alternatives growing)

#### Level 2 — Advanced
- **Site-to-site VPN in cloud**:
  - AWS: Virtual Private Gateway + Customer Gateway, BGP or static routing
  - Dual tunnels for redundancy (active/passive or active/active with ECMP)
  - Throughput: ~1.25 Gbps per tunnel (AWS), use multiple tunnels or Direct Connect for more
  - Latency: encrypted tunnel adds ~1-2ms overhead
  - Transit Gateway VPN: centralized VPN termination for multiple VPCs, supports ECMP
- **Client VPN architectures**:
  - Split tunneling: only VPN-destined traffic goes through the tunnel, internet traffic goes direct
  - Full tunnel: all traffic routes through VPN — better for security, worse for performance
  - Always-on VPN: forced VPN for compliance (Zscaler, Prisma Access, Cisco AnyConnect)
- **VPN vs Direct Connect / ExpressRoute / Cloud Interconnect**:
  - VPN: encrypted over public internet, variable latency, quick to set up
  - Dedicated connections: private fiber, consistent latency, higher bandwidth (1–100 Gbps), months to provision
  - Hybrid: VPN as backup for dedicated connections
- **VPN debugging**:
  - Phase 1 issues: mismatched IKE versions, encryption algorithms, PSK mismatch, NAT-T not enabled
  - Phase 2 issues: mismatched subnets (proxy IDs), PFS mismatch, transform set mismatch
  - Common: asymmetric routing, MTU issues (overhead reduces effective MTU), GRE/IPSec interaction
  - Tools: `ipsec statusall` (strongSwan), `show crypto isakmp sa` (Cisco), CloudWatch VPN metrics

### 3.11 Firewalls

#### Level 0 — Basics
- **Stateful firewall**: tracks connection state, automatically allows return traffic for established connections
- **Stateless firewall**: evaluates each packet independently, must explicitly allow both inbound and outbound
- **Common model**: default deny inbound, allow outbound, allow established/related return traffic

#### Level 1 — Intermediate
- **Security Groups (AWS/cloud)**:
  - Instance-level (ENI-level) firewall
  - Stateful: allow rule for inbound automatically permits matching outbound response
  - Allow rules only — no deny rules
  - Can reference other security groups as source/destination (powerful for microservice communication)
  - Changes take effect immediately
  - Default: deny all inbound, allow all outbound
  - Evaluation: all rules evaluated (union of all allow rules)
- **NACLs (Network ACLs)**:
  - Subnet-level firewall
  - Stateless: must explicitly allow both inbound and outbound (including ephemeral port ranges)
  - Allow AND deny rules
  - Rules processed in order by rule number (lowest first), first match wins
  - Default NACL: allow all inbound and outbound
  - Use case: broad subnet-level blocking (e.g., block a CIDR, rate-limit specific port)
- **Security Groups vs NACLs**:
  - SGs are first choice for granular access control (stateful, can reference SGs)
  - NACLs are defense in depth — subnet-level guardrails
  - Combined: NACL evaluated first → then security group

#### Level 2 — Advanced
- **iptables**:
  - **Tables**: `filter` (default, packet filtering), `nat` (address translation), `mangle` (packet modification), `raw` (connection tracking exemption)
  - **Chains**:
    - `INPUT`: packets destined for the local machine
    - `OUTPUT`: packets generated by the local machine
    - `FORWARD`: packets routed through the machine
    - `PREROUTING`: before routing decision (DNAT, mangle)
    - `POSTROUTING`: after routing decision (SNAT, masquerade)
  - **Targets**: `ACCEPT`, `DROP`, `REJECT`, `LOG`, `SNAT`, `DNAT`, `MASQUERADE`, `REDIRECT`
  - **Rule matching**: protocol, source/dest IP, source/dest port, interface, connection state (`-m conntrack --ctstate`)
  - **Packet flow**: PREROUTING → routing decision → FORWARD/INPUT → OUTPUT → POSTROUTING
  - **Common patterns**:
    - Allow established: `-A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT`
    - Allow SSH: `-A INPUT -p tcp --dport 22 -s 10.0.0.0/8 -j ACCEPT`
    - Default deny: `-P INPUT DROP`
    - Port forwarding: `-t nat -A PREROUTING -p tcp --dport 80 -j DNAT --to 10.0.0.5:8080`
  - **nftables**: successor to iptables, unified framework, better performance, atomic rule updates
- **Cloud firewalls beyond SG/NACL**:
  - AWS Network Firewall: managed IDS/IPS, Suricata-based, stateful inspection, domain filtering
  - Web Application Firewall (WAF): L7 protection, OWASP rules, rate limiting, bot management
  - Host-based firewalls: `firewalld`, `ufw`, Windows Firewall — defense in depth
- **Kubernetes network policies**:
  - Namespace-level, label-selector-based firewall rules
  - Ingress and egress rules for pods
  - Requires a CNI that supports network policies (Calico, Cilium, Antrea)
  - Default: all traffic allowed between pods if no network policy exists
  - Default deny policy: `podSelector: {}` with no ingress/egress rules
  - Limitation: no deny rules, no logging (use Cilium for L7 policies and observability)

### 3.12 Packet Flow

#### Level 0 — Basics
- **Basic packet flow**: Application → Socket → TCP/UDP → IP → NIC driver → Wire → NIC → IP → TCP/UDP → Socket → Application
- **Each hop**: TTL decremented, MAC addresses rewritten, routing decision made

#### Level 1 — Intermediate (Cloud Packet Flow)
- **Ingress to an EC2 instance via ALB**:
  1. Client DNS resolves ALB domain → ALB node IP (anycast or DNS round-robin)
  2. Client TCP handshake with ALB node
  3. TLS handshake (if HTTPS listener)
  4. ALB inspects HTTP request, evaluates rules (host, path, header matching)
  5. ALB selects target (target group, algorithm)
  6. ALB opens new TCP connection to backend (or reuses via connection pooling)
  7. NACL evaluates inbound on target subnet → Security group evaluates inbound on target ENI
  8. Packet reaches instance, response travels reverse path
  9. ALB inserts `X-Forwarded-For`, `X-Forwarded-Proto` headers
- **DNAT/SNAT in cloud**:
  - **Inbound DNAT**: ALB/NLB/IGW translates destination from public IP to private IP
  - **Outbound SNAT**: NAT gateway translates source from private IP to NAT gateway's Elastic IP
  - **Inter-VPC**: transit gateway or peering — no NAT, packets routed directly (if CIDR non-overlapping)
  - **VPC endpoint**: traffic routed internally through AWS backbone, no NAT

#### Level 2 — Advanced (Kubernetes Packet Flow)
- **Pod-to-pod (same node)**:
  1. Source pod sends packet via `eth0` (veth pair connected to bridge `cbr0` or CNI-managed interface)
  2. Bridge forwards to destination pod's veth pair
  3. No NAT involved
- **Pod-to-pod (different node)**:
  - **Overlay (VXLAN/GENEVE)**: packet encapsulated with outer node IP, sent to destination node, decapsulated
  - **Direct routing (BGP with Calico)**: pod CIDR routes advertised via BGP, no encapsulation overhead
  - **AWS VPC CNI**: pods get IPs from VPC subnet (ENI secondary IPs), native VPC routing — no overlay
- **External to pod via Service (NodePort/LoadBalancer)**:
  1. Client → cloud load balancer (NLB/ALB) → node IP:NodePort
  2. iptables/IPVS on node: DNAT from `NodeIP:NodePort` to `PodIP:PodPort`
  3. If pod is on a different node: packet forwarded to that node (SNAT with node IP as source)
  4. `externalTrafficPolicy: Local`: skip inter-node hop, preserve client IP, but traffic only goes to nodes with local pods
  5. Response: reverse DNAT/SNAT, back through LB to client
- **Service ClusterIP flow (iptables mode)**:
  1. Pod sends to ClusterIP:port
  2. iptables rules (set by kube-proxy) randomly DNAT to one of the endpoint pod IPs
  3. Conntrack ensures return traffic is un-DNATed correctly
  4. IPVS mode: replaces iptables rules with IPVS rules — O(1) lookup vs O(n) chain traversal
- **Packet capture analysis**:
  - `tcpdump -i eth0 -nn port 80` — capture HTTP traffic, raw IPs
  - `tcpdump -i any -w capture.pcap` — write to file for Wireshark analysis
  - `tcpdump -i eth0 'tcp[tcpflags] & (tcp-syn) != 0'` — capture only SYN packets
  - Wireshark filters: `tcp.stream eq 0`, `http.request.method == "GET"`, `tls.handshake.type == 1`
  - Capture on pod network namespace: `nsenter -t <pid> -n tcpdump -i eth0`

### 3.13 CDN (Content Delivery Network)

#### Level 0 — Basics
- **Purpose**: serve content from edge locations geographically closer to users, reducing latency
- **How it works**: client request → nearest edge (PoP) → cache hit (serve) or cache miss (fetch from origin)
- **Content types**: static (images, CSS, JS, fonts), dynamic (API responses with short TTL), streaming (video segments)
- **Providers**: CloudFront, Cloudflare, Akamai, Fastly, Azure CDN, Google Cloud CDN

#### Level 1 — Intermediate
- **Edge caching**:
  - Cache key: typically URL + headers specified in `Vary` (e.g., `Vary: Accept-Encoding`)
  - TTL control: origin `Cache-Control` headers vs CDN-level TTL overrides
  - Cache behaviors/rules: route patterns (e.g., `/api/*` → no cache, `/static/*` → cache 30d)
  - Compression: gzip/brotli at edge
- **Origin shield**:
  - Intermediate caching layer between edge PoPs and origin
  - Reduces origin load: multiple edge misses become a single origin request
  - Especially useful when origin is under heavy load or has limited capacity
  - CloudFront: Origin Shield, Cloudflare: Tiered Cache
- **Cache invalidation**:
  - Path-based invalidation: `/*`, `/images/*`, `/index.html`
  - Tag-based invalidation: Fastly surrogate keys, Cloudflare cache tags
  - Versioned URLs: `/app.v2.js` or `/app.js?v=abc123` — preferred over invalidation
  - Invalidation propagation delay: seconds to minutes depending on CDN
  - Cost: some CDNs charge per invalidation path (CloudFront: first 1000/month free)
- **TTLs in CDN**:
  - Browser TTL (`Cache-Control: max-age`) vs CDN TTL (`s-maxage`)
  - Stale-while-revalidate: serve stale content while fetching fresh copy
  - Stale-if-error: serve stale content if origin is unavailable
  - Minimum TTL: CDN enforces a floor regardless of origin headers

#### Level 2 — Advanced
- **CDN for dynamic content**:
  - API acceleration: persistent connections from edge to origin, optimized routes (e.g., Cloudflare Argo)
  - Edge compute: run logic at edge (CloudFront Functions, Cloudflare Workers, Fastly Compute@Edge)
  - A/B testing at edge, geo-based routing, header manipulation
- **CDN security**:
  - DDoS protection: absorb L3/L4/L7 attacks at edge
  - WAF integration: block malicious requests at edge before reaching origin
  - Bot management: challenge suspicious traffic
  - Origin shielding: restrict origin to accept connections only from CDN IPs
  - Signed URLs/cookies: restrict access to authorized users (time-limited, IP-limited)
- **CDN monitoring and observability**:
  - Cache hit ratio: percentage of requests served from cache (target: >90% for static)
  - Origin request rate: should be a fraction of edge request rate
  - Error rates: 4xx (client), 5xx (origin errors)
  - Real-time logs: stream to S3, Splunk, Datadog for analysis
  - Core Web Vitals: LCP, FID/INP, CLS — CDN impact on performance metrics

### 3.14 Service Mesh Networking

#### Level 0 — Basics
- **What a service mesh is**: dedicated infrastructure layer for service-to-service communication
- **Problem it solves**: consistent observability, security (mTLS), traffic management without application changes
- **Architecture**: data plane (sidecar proxies) + control plane (configuration, certificate management)
- **Popular implementations**: Istio, Linkerd, Consul Connect, Cilium (sidecar-less with eBPF)

#### Level 1 — Intermediate
- **Sidecar proxy**:
  - Injected alongside each application pod (Envoy for Istio, linkerd2-proxy for Linkerd)
  - Intercepts all inbound and outbound traffic via iptables rules (redirect to proxy port)
  - Handles: TLS termination/origination, retry, timeout, circuit breaking, load balancing
  - Transparent to the application — no code changes needed
  - Resource overhead: CPU and memory per sidecar, added latency (~1-2ms p99)
- **mTLS in service mesh**:
  - Automatic mutual TLS between all services in the mesh
  - Control plane acts as CA, issues short-lived certificates (SPIFFE identity)
  - Certificate rotation: automatic, transparent, typically every 24h
  - Strict mode: reject non-mTLS traffic; permissive mode: accept both (migration)
  - Replaces network-level trust with cryptographic identity
- **Traffic management**:
  - **VirtualService**: route rules (match on headers, URI, authority), retry policies, fault injection
  - **DestinationRule**: load balancing policy, connection pool settings, outlier detection
  - **Traffic splitting**: canary deployments (e.g., 95% v1, 5% v2), header-based routing for testing
  - **Retries**: automatic retry with configurable attempts, per-try timeout, retry conditions (5xx, connect-failure)
  - **Timeouts**: request-level and per-try timeouts — prevent cascading failures
  - **Circuit breaking**: limit concurrent connections/requests to a destination, shed load when overwhelmed

#### Level 2 — Advanced
- **Observability**:
  - Automatic metrics: request rate, error rate, latency (RED) per service pair — no instrumentation needed
  - Distributed tracing: inject trace headers (B3, W3C TraceContext), visualize request paths (Jaeger, Zipkin)
  - Access logging: structured logs for every request through the mesh
  - Service graph: visualize dependencies and traffic flow (Kiali for Istio)
- **Advanced traffic patterns**:
  - **Fault injection**: inject delays (latency testing) or aborts (error handling testing) at mesh level
  - **Traffic mirroring (shadowing)**: send copy of live traffic to a new version without affecting responses
  - **Rate limiting**: global and local rate limits at mesh level
  - **Locality-aware load balancing**: prefer same-zone backends to reduce cross-zone traffic cost
- **Sidecar-less mesh (ambient mesh / eBPF)**:
  - Istio ambient mesh: per-node ztunnel proxy (L4) + optional waypoint proxies (L7) — no sidecar
  - Cilium: eBPF-based mesh, handles mTLS and policy in kernel, lowest overhead
  - Tradeoffs: less resource overhead, but less per-pod isolation and control
- **Multi-cluster mesh**:
  - Extend mesh across multiple Kubernetes clusters
  - Shared trust domain: common root CA across clusters
  - Service discovery: federated endpoints, DNS-based or control plane sync
  - Traffic routing across clusters for DR, regional failover

### 3.15 Network Debugging Tools

#### Level 0 — Basics
- **ping**: ICMP echo — test reachability, measure RTT
  - `ping -c 4 10.0.0.1` — send 4 packets
  - Note: many cloud environments block ICMP — do not rely on ping alone
- **traceroute / tracepath**: show path packets take, identify where latency or drops occur
  - `traceroute -n 10.0.0.1` — numeric output (skip DNS resolution)
  - Uses incrementing TTL to discover each hop
  - `*` in output: hop does not respond to probes (firewall or ICMP rate limiting)
- **dig**: DNS query tool
  - `dig example.com A` — query A record
  - `dig @8.8.8.8 example.com` — query specific resolver
  - `dig +trace example.com` — iterative resolution from root
  - `dig +short example.com` — concise output
- **nslookup**: simpler DNS query tool (older, less flexible than dig)
  - `nslookup example.com 8.8.8.8`

#### Level 1 — Intermediate
- **tcpdump**: command-line packet capture
  - `tcpdump -i eth0 -nn host 10.0.0.5 and port 443` — filter by host and port, numeric output
  - `tcpdump -i eth0 -w capture.pcap` — write raw packets to file
  - `tcpdump -r capture.pcap` — read from file
  - `tcpdump -i eth0 'tcp[tcpflags] == tcp-syn'` — capture only SYN packets
  - `tcpdump -i eth0 -c 100 -s 0` — capture 100 full packets
  - BPF filters: `src 10.0.0.0/24`, `dst port 80`, `icmp`, `arp`
- **mtr (my traceroute)**: combines ping and traceroute — continuous monitoring with loss/latency per hop
  - `mtr -n --report 10.0.0.1` — generate report with 10 rounds
  - Identify: packet loss at a specific hop, latency spikes, asymmetric routing
  - Interpreting: loss at intermediate hop but not at destination = ICMP rate limiting (not real loss)
- **curl**: HTTP(S) client and debugging tool
  - `curl -v https://example.com` — verbose output with TLS handshake, headers
  - `curl -o /dev/null -s -w "%{time_total} %{http_code}\n" https://example.com` — measure total time
  - `curl -w "@curl-timing.txt" https://example.com` — detailed timing (dns, connect, tls, ttfb)
  - `curl -k https://example.com` — skip TLS verification (testing only)
  - `curl --resolve example.com:443:10.0.0.5 https://example.com` — override DNS resolution
  - `curl -H "Host: example.com" http://10.0.0.5` — test virtual host routing
- **openssl s_client**: TLS connection debugging
  - `openssl s_client -connect example.com:443 -servername example.com` — full TLS handshake info
  - `openssl s_client -connect example.com:443 -showcerts` — show full certificate chain
  - `openssl s_client -connect example.com:443 -tls1_2` — force TLS version
  - Check certificate expiry: `echo | openssl s_client -connect host:443 2>/dev/null | openssl x509 -noout -dates`
  - Check SANs: `echo | openssl s_client -connect host:443 2>/dev/null | openssl x509 -noout -ext subjectAltName`

#### Level 2 — Advanced
- **Wireshark**: GUI packet analyzer
  - Display filters: `tcp.stream eq 5`, `http.response.code == 500`, `tls.handshake.type == 1`, `dns.qry.name == "example.com"`
  - Follow TCP stream: reassemble conversation
  - I/O graphs: visualize traffic patterns, retransmissions
  - Expert info: highlights anomalies (retransmissions, zero windows, RST, duplicate ACKs)
  - TLS decryption: provide SSLKEYLOGFILE for session key logging (Firefox/Chrome support)
  - Kubernetes: capture on node and filter by pod IP, or use `ksniff` to capture directly on pod
- **ss (socket statistics)**: modern replacement for netstat
  - `ss -tlnp` — listening TCP sockets with process info
  - `ss -tanp state established` — established connections
  - `ss -s` — summary statistics (TCP states, memory)
  - `ss -tanp dst 10.0.0.5` — filter by destination
  - `ss -i` — internal TCP info (cwnd, rtt, retrans)
- **nmap**: network scanner
  - `nmap -sT -p 80,443 10.0.0.0/24` — TCP connect scan on specific ports
  - `nmap -sU -p 53 10.0.0.1` — UDP scan
  - `nmap -sV -p 1-1024 10.0.0.1` — service version detection
  - Use responsibly: only scan networks you own/manage
- **iperf3**: network throughput testing
  - Server: `iperf3 -s`
  - Client: `iperf3 -c 10.0.0.1 -t 30 -P 4` — 30s test with 4 parallel streams
  - Test: bandwidth, jitter, packet loss between two points
- **netcat (nc)**: TCP/UDP swiss army knife
  - `nc -zv 10.0.0.1 80` — test port connectivity
  - `nc -l -p 8080` — listen on port (simple server for testing)
  - `echo "test" | nc -u 10.0.0.1 514` — send UDP datagram

### 3.16 Failure Scenarios & Debugging — Networking

- **DNS resolution failure**:
  - Symptoms: `Could not resolve host`, `NXDOMAIN`, `SERVFAIL`
  - Debug: `dig +trace domain.com`, check `/etc/resolv.conf`, check if DNS server is reachable (`nc -zv dns-server 53`)
  - Causes: incorrect nameserver config, DNS server overloaded, missing DNS records, DNSSEC validation failure, CoreDNS pods down (Kubernetes)
  - Fix: verify DNS records (`dig @authoritative-ns domain.com`), check DNS pod logs, flush caches

- **Connection timeout**:
  - Symptoms: `Connection timed out`, SYN sent but no SYN-ACK received
  - Debug: `tcpdump -i eth0 host <target> and port <port>` — look for SYN without SYN-ACK
  - Causes: firewall dropping packets (security group, NACL, iptables), wrong route, target not listening, network path down
  - Fix: check security groups, NACLs, route tables, verify service is listening (`ss -tlnp`)

- **Connection refused**:
  - Symptoms: immediate `RST` in response to `SYN`
  - Debug: `tcpdump` shows RST, `ss -tlnp` on target shows no listener on that port
  - Causes: service not running, listening on wrong interface (127.0.0.1 vs 0.0.0.0), wrong port
  - Fix: start the service, fix bind address, check port configuration

- **TLS handshake failure**:
  - Symptoms: `SSL_ERROR_HANDSHAKE_FAILURE`, `certificate verify failed`, `unknown_ca`
  - Debug: `openssl s_client -connect host:443 -servername host` — inspect certificate chain, cipher negotiation
  - Causes: expired certificate, untrusted CA, missing intermediate cert, SNI mismatch, cipher suite mismatch, clock skew
  - Fix: renew cert, install intermediate certs, verify hostname matches SAN/CN, enable compatible cipher suites, sync NTP

- **Intermittent packet loss**:
  - Symptoms: retransmissions, high latency variance, sporadic timeouts
  - Debug: `mtr -n --report target` — identify lossy hop; `ss -i` — check retransmits
  - Causes: network congestion, NIC errors (`ethtool -S eth0`), buffer overflow (`netstat -s | grep overflow`), faulty hardware
  - Fix: check interface errors, increase buffer sizes, enable TCP BBR, investigate congestion at identified hop

- **Conntrack table exhaustion**:
  - Symptoms: random packet drops, `nf_conntrack: table full, dropping packet` in `dmesg`
  - Debug: `cat /proc/sys/net/netfilter/nf_conntrack_count` vs `nf_conntrack_max`, `conntrack -L | wc -l`
  - Causes: high connection rate, long timeouts, many TIME_WAIT connections, DDoS
  - Fix: increase `nf_conntrack_max`, reduce timeouts (`nf_conntrack_tcp_timeout_established`), enable `tcp_tw_reuse`

- **MTU / fragmentation issues (black hole)**:
  - Symptoms: small packets work, large packets fail; TCP handshake succeeds but data transfer stalls
  - Debug: `ping -M do -s 1472 target` (test 1500 MTU), `tcpdump` for ICMP "Fragmentation Needed"
  - Causes: PMTUD broken (firewall blocking ICMP), tunnel overhead reducing effective MTU (IPSec, VXLAN)
  - Fix: reduce MTU on interface, set TCP MSS clamping (`iptables -A FORWARD -p tcp --tcp-flags SYN,RST SYN -j TCPMSS --clamp-mss-to-pmtu`)

- **Asymmetric routing**:
  - Symptoms: traffic goes out one path, returns via another; stateful firewalls drop return traffic
  - Debug: `traceroute` from both ends, check route tables for mismatched paths
  - Causes: multiple default gateways, misconfigured routing, multi-homed hosts without policy routing
  - Fix: ensure symmetric paths, use policy-based routing (`ip rule`), or adjust firewall to handle asymmetry

- **Port exhaustion**:
  - Symptoms: `Cannot assign requested address`, connection failures for new outbound connections
  - Debug: `ss -tan | awk '{print $4}' | sort | uniq -c | sort -rn` — count connections per source port range
  - Causes: ephemeral port range exhausted, too many TIME_WAIT connections, NAT gateway port limits
  - Fix: increase ephemeral port range (`net.ipv4.ip_local_port_range`), enable `tcp_tw_reuse`, use connection pooling, add NAT gateway IPs

- **CLOSE_WAIT accumulation**:
  - Symptoms: growing number of `CLOSE_WAIT` connections, eventual resource exhaustion
  - Debug: `ss -tanp state close-wait` — identify the process holding connections
  - Causes: application not calling `close()` on sockets after receiving `FIN` — a code bug
  - Fix: fix application code, implement proper connection cleanup, connection pool tuning

- **502 Bad Gateway / 504 Gateway Timeout**:
  - Symptoms: load balancer returns 502 or 504
  - Debug: check LB access logs, check backend health checks, check backend application logs
  - 502 causes: backend closed connection prematurely, backend returned invalid response, backend health check failing
  - 504 causes: backend did not respond within timeout, network partition between LB and backend
  - Fix: increase backend timeout, fix backend health, ensure keep-alive alignment between LB and backend

- **Kubernetes DNS issues**:
  - Symptoms: pods cannot resolve service names, `nslookup kubernetes.default` fails
  - Debug: check CoreDNS pods (`kubectl get pods -n kube-system`), check CoreDNS logs, check `/etc/resolv.conf` in pod
  - Causes: CoreDNS crashed/OOM, too many DNS queries (ndots:5 causing unnecessary lookups), upstream DNS failure
  - Fix: restart CoreDNS, tune ndots, add FQDN (trailing dot), increase CoreDNS replicas, check CoreDNS ConfigMap

### 3.17 Interview Questions — Networking

1. Walk through what happens at the network level when you type `https://example.com` in a browser and press Enter. Cover DNS, TCP, TLS, and HTTP.
2. Explain the TCP 3-way handshake and 4-way teardown. Why does `TIME_WAIT` exist, and what problems can excessive `TIME_WAIT` connections cause?
3. What is the difference between TCP flow control and congestion control? Explain the mechanisms used for each.
4. Compare HTTP/1.1, HTTP/2, and HTTP/3. What specific problems does each newer version solve?
5. Describe the TLS 1.3 handshake. How does it differ from TLS 1.2, and why is it faster?
6. What is mutual TLS (mTLS)? Where and why would you use it? How does a service mesh implement it?
7. Explain the difference between L4 and L7 load balancing. When would you choose one over the other?
8. A service behind an AWS ALB is returning 502 errors intermittently. Walk through your debugging approach step by step.
9. How does a packet travel from an external client to a pod inside a Kubernetes cluster via a LoadBalancer service? Include all NAT translations and routing decisions.
10. Explain VPC peering vs Transit Gateway vs PrivateLink. When would you use each?
11. What is the conntrack table, and what happens when it is exhausted? How would you detect and resolve this?
12. You notice that DNS resolution is slow in your Kubernetes pods (adding 5s delay). How would you debug this? What is the `ndots` issue?
13. Explain how `iptables` processes a packet — tables, chains, and rule evaluation order. How does Kubernetes use `iptables` for Service routing?
14. What is BGP, and why is it important in cloud networking? How does it relate to Direct Connect or Cloud Interconnect?
15. Describe split-horizon DNS. Why is it used, and how would you implement it in a cloud environment?
16. What is the difference between a stateful and stateless firewall? Compare Security Groups and NACLs in AWS.
17. Explain CDN cache invalidation strategies. What is origin shield, and when would you use it?
18. A server can accept new TCP connections but data transfer is extremely slow. What could be wrong? Walk through your investigation.

### 3.18 Hands-on Labs — Networking

- [ ] Capture a full TCP 3-way handshake and 4-way teardown with `tcpdump` and analyze in Wireshark — identify SYN, SYN-ACK, ACK, FIN, and window sizes
- [ ] Set up a VPC with public and private subnets, NAT gateway, internet gateway, and proper route tables — verify connectivity from private subnet to internet
- [ ] Configure a split-horizon DNS setup using BIND or CoreDNS — return different IPs based on client source network
- [ ] Set up an nginx reverse proxy with TLS termination, generate certificates with `openssl`, and verify the full certificate chain using `openssl s_client`
- [ ] Build iptables rules for a multi-tier application: web tier (allow 80/443 from anywhere), app tier (allow 8080 from web tier only), DB tier (allow 5432 from app tier only) — test and verify
- [ ] Deploy a Kubernetes service with ClusterIP, NodePort, and LoadBalancer types — trace the full packet path from external client to pod using `tcpdump` on the node
- [ ] Simulate and debug conntrack table exhaustion: create many short-lived connections, observe `nf_conntrack: table full` in dmesg, tune parameters to resolve
- [ ] Set up a WireGuard VPN tunnel between two machines — verify encrypted traffic with `tcpdump`, test routing and DNS resolution through the tunnel
- [ ] Configure HAProxy with L7 load balancing, health checks, sticky sessions, and connection draining — perform a rolling deployment and verify zero dropped requests
- [ ] Use `mtr` and `traceroute` to identify a network bottleneck — simulate packet loss with `tc netem` and measure impact on TCP throughput with `iperf3`
- [ ] Set up mTLS between two services manually using `openssl` — generate CA, server cert, client cert, and verify mutual authentication
- [ ] Deploy Istio service mesh on a Kubernetes cluster — observe automatic mTLS, configure traffic splitting (canary), inject faults, and view service graph in Kiali
- [ ] Configure CloudFront (or Cloudflare) CDN with origin shield, cache behaviors for static and dynamic content, signed URLs, and verify cache hit/miss ratios
- [ ] Debug a simulated DNS failure: misconfigure `/etc/resolv.conf`, observe timeout behavior, use `dig +trace` to identify the issue, and fix it
- [ ] Analyze HTTP/2 multiplexing vs HTTP/1.1 performance: use `curl --http2` and `curl --http1.1` with timing, capture with Wireshark, compare stream handling

---

## Section 4 — AWS

### 4.1 AWS Fundamentals

#### AWS Global Infrastructure
- **Regions**: geographic areas with 2+ AZs (e.g., us-east-1, eu-west-1), choose based on latency, compliance, service availability, cost
- **Availability Zones (AZs)**: isolated data centers within a region, connected via low-latency links, design for multi-AZ
- **Edge locations**: CloudFront PoPs, Route 53, Global Accelerator — content caching and DNS closer to users
- **Local Zones**: AWS infrastructure extensions closer to end users (e.g., Los Angeles, Boston)
- **Wavelength Zones**: AWS at telecom carrier edge for ultra-low latency (5G)

#### AWS Account Structure
- **AWS Organizations**: management account + member accounts, consolidated billing
- **Organizational Units (OUs)**: group accounts (e.g., Prod, Dev, Security, Sandbox)
- **Service Control Policies (SCPs)**: guardrails applied to OUs/accounts, deny-list or allow-list approach
- **Control Tower**: automated landing zone setup, account factory, guardrails
- **Multi-account strategy**: account-per-environment, account-per-team, shared services account

### 4.2 IAM Deep Dive

#### Level 0 — Basics
- **Users**: individual identities, access keys + secret keys (avoid long-lived credentials)
- **Groups**: collection of users, attach policies to groups not users
- **Roles**: assumable identities, no permanent credentials, used by services/cross-account/federation
- **Policies**: JSON documents defining permissions (Effect, Action, Resource, Condition)

#### Level 1 — Intermediate
- **Policy types**:
  - Identity-based: attached to users/groups/roles
  - Resource-based: attached to resources (S3 bucket policy, SQS policy, KMS key policy)
  - Permission boundaries: maximum permissions an identity can have
  - SCPs: maximum permissions for accounts in an organization
  - Session policies: limit permissions for assumed role sessions
- **Policy evaluation logic**: explicit Deny → SCP → permission boundary → identity policy → resource policy
- **Trust policies**: who can assume a role (principal, condition), cross-account trust
- **Condition keys**: `aws:SourceIp`, `aws:PrincipalOrgID`, `aws:RequestedRegion`, `aws:MultiFactorAuthPresent`
- **IAM Access Analyzer**: identify resources shared externally, unused access, policy validation

#### Level 2 — Advanced
- **IRSA (IAM Roles for Service Accounts)**: EKS pods assume IAM roles via OIDC federation
  - EKS cluster has OIDC provider → service account annotated with role ARN → pod gets temporary credentials via STS
  - Trust policy: `"Principal": {"Federated": "arn:aws:iam::oidc-provider/oidc.eks..."}`
  - Condition: `"StringEquals": {"oidc.eks....:sub": "system:serviceaccount:namespace:sa-name"}`
- **EKS Pod Identity**: newer alternative, EKS Pod Identity Agent DaemonSet, simpler setup than IRSA
- **Cross-account access**: role assumption flow — caller calls STS AssumeRole → gets temporary credentials → accesses resources in target account
- **Federation**: SAML 2.0, OIDC (GitHub Actions, Google), AWS SSO/Identity Center
- **Permission boundaries**: cap permissions for delegated admin (dev team can create roles but only within boundary)
- **Session tags**: pass metadata during role assumption, use in policy conditions

### 4.3 Compute — EC2

#### Instance Types
- **General purpose**: M-series (m6i, m7g ARM) — balanced compute, memory, networking
- **Compute optimized**: C-series (c6i, c7g) — high-performance computing, batch
- **Memory optimized**: R-series (r6i), X-series — in-memory databases, caches
- **Storage optimized**: I-series, D-series — high sequential I/O, data warehousing
- **Accelerated/GPU**:
  - P4d/P4de (A100) — ML training, 8x A100 40/80GB, 400 Gbps EFA
  - P5 (H100) — latest ML training, 8x H100 80GB, 3200 Gbps EFA
  - G5 (A10G) — inference, graphics, 1-8 GPUs
  - G6 (L4) — cost-effective inference
  - Inf2 (Inferentia2) — optimized inference, cost-effective for transformer models
  - Trn1 (Trainium) — optimized training, cost-effective alternative to GPU

#### Purchasing Options
- **On-Demand**: pay per second, no commitment, highest cost
- **Reserved Instances (RI)**: 1 or 3-year commitment, up to 72% savings, standard vs convertible
- **Savings Plans**: flexible commitment (compute or EC2), applies across instance families/regions
- **Spot Instances**: up to 90% savings, can be interrupted with 2-min notice, use for fault-tolerant workloads
- **Dedicated Hosts/Instances**: physical server dedication, compliance, licensing requirements

#### Advanced EC2
- **Nitro system**: custom hardware (Nitro cards), hypervisor offloading, enhanced security, bare-metal performance
- **IMDSv2**: instance metadata service v2, requires session token (PUT request first), mitigates SSRF attacks
- **ENA (Elastic Network Adapter)**: enhanced networking, up to 100 Gbps
- **EFA (Elastic Fabric Adapter)**: RDMA-like networking for HPC/ML, bypasses kernel networking stack, required for multi-node training with NCCL
- **Placement groups**:
  - Cluster: same rack, lowest latency, HPC/ML training
  - Spread: different racks, max 7 per AZ, high availability
  - Partition: logical partitions on separate racks, large distributed systems (Kafka, HDFS)
- **Auto Scaling Groups**: launch templates, scaling policies (target tracking, step, simple), lifecycle hooks, warm pools, predictive scaling, mixed instances policy (spot + on-demand)

### 4.4 Networking — VPC

#### VPC Design
- **CIDR planning**: avoid overlapping CIDRs across VPCs, plan for future growth, /16 for production VPCs
- **Subnet strategy**: 3 tiers × 3 AZs = 9 subnets (public, private, isolated/data)
  - Public: ALB, NAT gateway, bastion (if needed)
  - Private: application workloads, EKS nodes
  - Isolated/Data: RDS, ElastiCache, no internet access
- **Secondary CIDRs**: add additional CIDR blocks to VPC for pod networking (100.64.0.0/16)

#### Route Tables
- **Main route table**: default for subnets not explicitly associated
- **Custom route tables**: public (0.0.0.0/0 → IGW), private (0.0.0.0/0 → NAT GW), isolated (no default route)
- **Route priority**: most specific route wins (longest prefix match)
- **Route propagation**: automatically add routes from VPN/Direct Connect via Virtual Private Gateway

#### Gateways and Endpoints
- **Internet Gateway (IGW)**: 1:1 NAT for instances with public IPs, highly available
- **NAT Gateway**: managed SNAT, per-AZ deployment, 55K concurrent connections per destination, $0.045/GB processing
- **VPC Endpoints**:
  - Gateway endpoints: S3, DynamoDB — free, route table entry
  - Interface endpoints (PrivateLink): ENI with private IP, per-hour + per-GB cost, DNS override
- **Transit Gateway**: hub-and-spoke, transitive routing, route table segmentation, inter-region peering, scales to thousands of VPCs
- **VPC Peering**: direct connection, non-transitive, no overlapping CIDRs, cross-region supported

#### Security
- **Security Groups**: stateful, ENI-level, allow rules only, reference other SGs
- **NACLs**: stateless, subnet-level, allow + deny, numbered rules (lowest first wins)
- **VPC Flow Logs**: metadata capture (src/dst IP, ports, action), CloudWatch/S3/Kinesis destinations
- **Traffic Mirroring**: full packet capture, sent to NLB target for analysis

### 4.5 Storage

#### S3
- **Storage classes**: Standard, Intelligent-Tiering, Standard-IA, One Zone-IA, Glacier Instant/Flexible/Deep Archive
- **Lifecycle policies**: transition objects between classes, expire objects, abort incomplete multipart uploads
- **Versioning**: protect against accidental deletes, MFA delete for extra protection
- **Replication**: CRR (cross-region), SRR (same-region), replication time control (15 min SLA)
- **Encryption**: SSE-S3 (default), SSE-KMS (audit trail, key rotation), SSE-C (customer-managed keys)
- **Bucket policies**: resource-based, condition keys (VPC endpoint, IP, SSL, org ID)
- **Access points**: named network endpoints, per-application access policies, VPC-restricted
- **Performance**: 5,500 GET/s per prefix, 3,500 PUT/s per prefix, multipart upload (>100MB), transfer acceleration, S3 Select

#### EBS
- **Volume types**:
  - gp3: 3000 IOPS baseline, up to 16K IOPS, 125 MB/s baseline, up to 1000 MB/s — default choice
  - io2 Block Express: up to 256K IOPS, mission-critical, multi-attach
  - st1: throughput optimized HDD, sequential workloads
  - sc1: cold HDD, infrequent access
- **Snapshots**: incremental, cross-region copy, encryption, fast snapshot restore
- **Encryption**: AES-256, KMS key, encrypt at creation or copy

#### EFS & FSx
- **EFS**: managed NFS, multi-AZ, elastic capacity, performance modes (general/max I/O), throughput modes (bursting/provisioned/elastic)
- **FSx for Lustre**: high-performance parallel filesystem, ML training data, S3 integration, 100+ GB/s throughput
- **FSx for NetApp ONTAP**: enterprise NAS, multi-protocol (NFS/SMB/iSCSI)

### 4.6 EKS (Elastic Kubernetes Service)

#### Architecture
- **Managed control plane**: AWS manages API server, etcd, scheduler, controller-manager across 3 AZs
- **Data plane options**:
  - Managed node groups: AWS manages EC2 instances, AMI updates, rolling upgrades
  - Self-managed nodes: full control, custom AMIs, launch templates
  - Fargate: serverless pods, per-pod VM isolation, no node management
- **EKS versions**: follows upstream K8s, ~14 months support per version, extended support available

#### EKS Networking
- **VPC CNI**: pods get IPs from VPC subnet (secondary IPs on ENIs)
  - Prefix delegation: assign /28 prefix per ENI slot, 16 IPs per slot, higher pod density
  - WARM_ENI_TARGET, WARM_IP_TARGET, MINIMUM_IP_TARGET tuning
  - Custom networking: pods in different subnets/CIDRs than nodes
- **Security groups for pods**: assign SGs directly to pods (branch ENIs)
- **Pod networking limits**: depends on instance type (ENIs × IPs per ENI), prefix delegation increases limits significantly

#### EKS IAM
- **IRSA**: OIDC provider → service account annotation → STS AssumeRoleWithWebIdentity
- **Pod Identity**: EKS Pod Identity Agent, simpler association (no OIDC provider management)
- **aws-auth ConfigMap**: maps IAM roles/users to Kubernetes RBAC (being replaced by access entries)
- **EKS access entries**: native API for IAM → K8s RBAC mapping

#### EKS Operations
- **Upgrades**: control plane upgrade (in-place) → add-on upgrade → node group rolling update → validate
- **Add-ons**: CoreDNS, kube-proxy, VPC CNI, EBS CSI, managed via EKS API or Terraform
- **Karpenter**: node lifecycle management, consolidation, disruption, provisioner/NodePool + EC2NodeClass, spot + on-demand mix, GPU support
- **Cluster Autoscaler**: ASG-based scaling, less efficient than Karpenter, still used in some setups
- **Velero**: cluster backup/restore, PV snapshots, scheduled backups, cross-cluster migration

#### EKS Security
- **RBAC**: Roles/ClusterRoles mapped from IAM via aws-auth or access entries
- **Network policies**: Calico or Cilium CNI plugin required
- **Secrets encryption**: AWS KMS envelope encryption for K8s secrets at rest
- **Pod Security Standards**: enforce via Pod Security Admission or OPA/Gatekeeper/Kyverno
- **Private endpoint**: API server endpoint in VPC, disable public access

### 4.7 Databases

#### RDS
- **Multi-AZ**: synchronous replication, automatic failover (60-120s), DNS endpoint failover
- **Read replicas**: async replication, up to 15 (Aurora), cross-region supported
- **Parameter groups**: engine configuration, apply immediately or during maintenance window
- **Backup**: automated daily snapshots (up to 35 days retention), manual snapshots, PITR

#### Aurora
- **Storage**: distributed across 3 AZs, 6 copies, auto-repairs, auto-grows up to 128 TB
- **Endpoints**: writer (primary), reader (load-balanced), custom (specific instances)
- **Aurora Serverless v2**: auto-scales ACUs (0.5-128), scales in seconds, pay per ACU-second
- **Global databases**: 1 primary region + up to 5 secondary, <1s replication lag, cross-region failover

#### DynamoDB
- **Data model**: partition key (hash), optional sort key (range), items up to 400 KB
- **Indexes**: GSI (alternate partition + sort key, eventually consistent), LSI (alternate sort key, strongly consistent)
- **Capacity modes**: on-demand (pay per request), provisioned (with auto-scaling)
- **DAX**: in-memory cache, microsecond reads, write-through
- **Streams**: ordered change log, event-driven processing with Lambda
- **Global tables**: multi-region, multi-active, automatic replication

### 4.8 Security Services
- **KMS**: CMKs, automatic key rotation (yearly), envelope encryption, key policies, grants
- **Secrets Manager**: automatic rotation, RDS/Redshift/DocumentDB integration, cross-account access
- **ACM**: free public TLS certificates, auto-renewal, integrated with ALB/CloudFront/API Gateway
- **WAF**: web ACLs, rate-based rules, IP set rules, managed rule groups (OWASP), per-request cost
- **Shield**: Standard (free, L3/L4 DDoS) vs Advanced ($3K/mo, L7 DDoS, DRT, cost protection)
- **GuardDuty**: threat detection, ML-based anomaly detection, EKS audit log monitoring, S3 protection
- **Security Hub**: centralized security findings, CIS/PCI compliance checks, aggregation across accounts
- **CloudTrail**: API call logging, management events (free), data events (per-event cost), Insights

### 4.9 Cost Optimization
- **Compute**: Savings Plans (most flexible), RIs (specific), Spot (fault-tolerant), Compute Optimizer recommendations
- **Storage**: S3 Intelligent-Tiering, lifecycle policies, EBS gp3 migration, delete unused volumes/snapshots
- **Networking**: VPC endpoints (avoid NAT GW processing for S3/DynamoDB), minimize cross-AZ traffic, CloudFront for S3
- **Data transfer costs**: intra-AZ free, inter-AZ $0.01/GB, NAT GW $0.045/GB, internet egress tiered
- **Tagging**: mandatory tags (team, environment, service, cost-center), AWS Cost Allocation Tags, tag enforcement via SCP/Config
- **Tools**: Cost Explorer, Budgets, Cost Anomaly Detection, Trusted Advisor, Compute Optimizer

### 4.10 Production Architecture Patterns
- **Multi-AZ HA**: resources spread across 3 AZs, ALB health checks, RDS Multi-AZ, EKS multi-AZ node groups
- **Multi-region DR**:
  - Backup/restore: RPO hours, RTO hours, lowest cost
  - Pilot light: core services running, scale up on failover
  - Warm standby: scaled-down copy, quick scale-up
  - Active-active: full deployment both regions, Route 53 latency routing
- **Landing zone**: Control Tower, account factory, guardrails, shared services VPC, log archive account, audit account
- **Hub-spoke networking**: Transit Gateway hub, VPC spokes, shared services VPC, inspection VPC with AWS Network Firewall

### 4.11 Failure Scenarios & Debugging — AWS
- **EC2 unreachable**: check SG inbound rules, NACL rules, route table, public IP/EIP, key pair, instance status checks
- **EKS pods no internet**: check NAT gateway exists, private subnet route table has 0.0.0.0/0 → NAT GW, VPC CNI ENI limits
- **S3 access denied**: check bucket policy, IAM policy, ACL, KMS key policy (if SSE-KMS), VPC endpoint policy, object ownership
- **RDS connection timeout**: check SG allows app port, RDS in same VPC or peered, DNS resolution of endpoint, `max_connections`
- **IRSA not working**: verify OIDC provider configured, trust policy matches SA namespace:name, SA annotated with role ARN, pod restarted after annotation
- **High bill**: Cost Explorer → filter by service, check data transfer, NAT gateway processing, unused resources, over-provisioned instances
- **Cross-account access**: check trust policy in target account, caller has sts:AssumeRole permission, external ID if required

### 4.12 Interview Questions — AWS
- Explain VPC architecture for a production EKS cluster.
- How does IRSA work internally? Walk through the token exchange flow.
- Design a multi-region active-active architecture on AWS.
- What happens when an AZ goes down in your EKS cluster?
- How do you optimize data transfer costs in AWS?
- Explain the difference between VPC peering and Transit Gateway.
- How does S3 achieve 11 9s of durability?
- Design a secure, cost-effective infrastructure for ML inference on EKS.
- How do you handle secrets in EKS workloads?
- EBS vs EFS vs S3 — when use each?
- How does Karpenter differ from Cluster Autoscaler?
- Security groups vs NACLs — differences and when to use each?
- Explain the AWS Nitro system.
- Design a landing zone for a multi-account AWS org.
- Explain cross-account IAM role assumption flow.
- How do you handle EKS upgrades with zero downtime?
- What is prefix delegation in VPC CNI and why use it?

### 4.13 Hands-on Labs — AWS
- [ ] Design and deploy production VPC with public/private/isolated subnets across 3 AZs
- [ ] Set up EKS cluster with managed node groups, IRSA, and Karpenter
- [ ] Configure VPC CNI with prefix delegation for high pod density
- [ ] Implement cross-account access with IAM roles
- [ ] Set up S3 replication with SSE-KMS encryption
- [ ] Deploy Aurora with read replicas and test failover
- [ ] Configure VPC endpoints for S3 and ECR
- [ ] Set up Transit Gateway connecting 3 VPCs
- [ ] Deploy GPU instances (P4/G5) with EFA for ML workloads
- [ ] Implement cost monitoring with Budgets and Cost Anomaly Detection

---

## Section 5 — Containers

### 5.1 Container Fundamentals
- **What containers are**: process isolation using Linux namespaces + cgroups + chroot/pivot_root — NOT lightweight VMs
- **Containers vs VMs**: shared kernel, no guest OS overhead, millisecond startup, higher density, weaker isolation
- **OCI specification**:
  - runtime-spec: how to run a container (config.json, lifecycle)
  - image-spec: image format (manifest, layers, config)
  - distribution-spec: how to push/pull images (registry API)
- **Container runtimes hierarchy**:
  - High-level: containerd, CRI-O — image management, container lifecycle, CRI implementation
  - Shim: containerd-shim — keeps container running after daemon restart
  - Low-level: runc (OCI reference), crun (C, faster), gVisor (application kernel), Kata (micro-VM)

### 5.2 Docker Deep Dive

#### Architecture
- **Docker daemon** (dockerd): manages images, containers, networks, volumes
- **containerd**: container lifecycle management, pulled out of Docker
- **runc**: OCI-compliant runtime, creates/runs containers
- **Docker CLI**: client communicating with daemon via REST API over Unix socket

#### Image Layers
- **UnionFS/OverlayFS**: stack of read-only layers + one read-write layer on top
- **Copy-on-write**: reads go through layers top-down, writes copy file to upper layer
- **Layer caching**: each Dockerfile instruction creates a layer, cache invalidated when instruction or context changes
- **Content-addressable storage**: layers identified by SHA256 digest

#### Dockerfile Best Practices
- **Multi-stage builds**: separate build and runtime stages, copy only artifacts needed
- **Layer ordering**: least-changing layers first (base image → deps → app code)
- **.dockerignore**: exclude unnecessary files from build context
- **Minimal base images**: distroless (Google), alpine (5MB), scratch (0 bytes, static binaries only)
- **No root**: USER directive to run as non-root, specify numeric UID
- **Single process**: one process per container, PID 1 responsibilities (signal handling, zombie reaping)
- **ENTRYPOINT vs CMD**: ENTRYPOINT = executable, CMD = default arguments; combined form preferred

#### Build Optimization
- **BuildKit**: parallel builds, secret mounts, SSH mounts, cache mounts, inline cache
- **buildx**: multi-platform builds (linux/amd64, linux/arm64), remote builders, build cache export
- **Cache mounts**: `--mount=type=cache,target=/root/.cache/pip` — persistent cache across builds
- **Multi-platform**: `docker buildx build --platform linux/amd64,linux/arm64` → manifest list

### 5.3 Container Networking
- **Bridge (default)**: docker0 bridge, veth pairs, containers communicate via bridge
  - Port mapping: `-p 8080:80` → iptables DNAT rule
  - Container DNS: embedded DNS server at 127.0.0.11
- **Host**: container shares host network namespace, no isolation, best performance
- **None**: no networking, isolated container
- **Overlay**: multi-host networking (Docker Swarm), VXLAN encapsulation
- **Macvlan**: container gets MAC address on physical network, appears as physical device
- **How bridge networking works**:
  1. Container gets veth pair (one end in container as eth0, other end on bridge)
  2. Bridge connects all container veth endpoints
  3. Outbound: MASQUERADE rule on host (SNAT)
  4. Inbound: DNAT rule for port mapping

### 5.4 Container Storage
- **OverlayFS internals**:
  - Lower layers: read-only image layers
  - Upper layer: read-write container layer
  - Merged view: unified filesystem visible to container
  - Whiteout files: mark deleted files from lower layers
  - Opaque directories: hide lower layer directory contents
- **Volumes**: managed by Docker, stored in /var/lib/docker/volumes/, survive container removal
- **Bind mounts**: map host path into container, full host path control
- **tmpfs**: in-memory filesystem, sensitive data, no persistence
- **Storage drivers**: overlay2 (default, recommended), devicemapper (legacy), btrfs, zfs

### 5.5 Container Security

#### Image Security
- **Base image selection**: official images, minimal (distroless/alpine), pin digest not tag
- **Vulnerability scanning**: Trivy, Grype, Snyk, ECR native scanning — scan in CI before push
- **Image signing**: cosign (Sigstore), Notary v2 — verify image provenance
- **SBOM**: Software Bill of Materials, syft for generation, track dependencies
- **Supply chain**: SLSA framework, provenance attestation, reproducible builds

#### Runtime Security
- **Read-only root**: `--read-only` flag, write to tmpfs or volumes only
- **Drop capabilities**: `--cap-drop ALL --cap-add NET_BIND_SERVICE` — minimal capabilities
- **No-new-privileges**: `--security-opt no-new-privileges` — prevent privilege escalation
- **Seccomp profiles**: restrict syscalls, default profile blocks ~44 dangerous syscalls
- **AppArmor/SELinux**: mandatory access control, confine container processes
- **User namespaces**: map container root to unprivileged host user, rootless containers

#### Secrets Handling
- **Never in images**: no secrets in Dockerfile (ENV, COPY), no secrets in layers
- **BuildKit secrets**: `--mount=type=secret` — not persisted in image layers
- **Runtime injection**: environment variables (less secure), mounted files, secret stores (Vault, Secrets Manager)

### 5.6 Container Registries
- **ECR**: AWS managed, lifecycle policies (expire untagged after N days), image scanning, cross-account access via resource policy, replication (cross-region, cross-account)
- **Registry internals**: manifests (describe image), blobs (layer content), tags (mutable pointers), digests (immutable identifiers)
- **Image garbage collection**: delete untagged images, lifecycle policies, tag immutability
- **OCI artifacts**: store non-container artifacts (Helm charts, WASM, signatures) in OCI registries

### 5.7 Advanced Container Patterns
- **Init containers**: run before app containers, initialization tasks, dependency checks
- **Sidecar containers**: logging agents, proxies, monitoring — run alongside app container
- **Ambassador pattern**: proxy container handling external service communication
- **Adapter pattern**: normalize output from app container (log format conversion)
- **Distroless**: no shell, no package manager, reduced attack surface, debugging with ephemeral containers
- **Scratch**: empty base image, only for static binaries (Go), smallest possible image

### 5.8 Failure Scenarios & Debugging — Containers
- **Container won't start**: check entrypoint/cmd, missing binaries/libraries, permission errors, read `docker logs`
- **OOM killed**: memory limit too low, memory leak in app, check `docker inspect` for OOMKilled flag, increase limits or fix app
- **Network issues**: DNS resolution (`docker exec dig`), iptables rules, bridge misconfigured, port conflicts
- **Image pull failures**: auth issues (docker login), registry rate limiting (Docker Hub), wrong tag/digest, network connectivity
- **Build failures**: cache invalidation, platform mismatch (amd64 vs arm64), BuildKit compatibility
- **Disk full**: dangling images (`docker image prune`), unused volumes (`docker volume prune`), build cache (`docker builder prune`)
- **Debugging tools**: `docker logs`, `docker exec -it sh`, `docker inspect`, `docker stats`, `docker events`, `crictl` (for containerd), `nsenter -t <pid> -n` (network namespace)

### 5.9 Interview Questions — Containers
- How do containers achieve isolation? Explain namespaces and cgroups.
- What is the difference between ENTRYPOINT and CMD?
- Explain Docker image layers and copy-on-write.
- How does container networking work on a single host?
- Walk through your container image hardening process.
- Explain the difference between containerd and runc.
- How would you debug a container that keeps OOM-killing?
- What is a multi-stage build and why use it?
- Explain the OCI specification and its components.
- How do you handle secrets in containers?
- What is the difference between a volume and a bind mount?
- Explain rootless containers and their security benefits.
- How does BuildKit differ from legacy Docker build?
- What is OverlayFS and how does copy-on-write work?
- How would you reduce a 1GB Docker image to under 100MB?
- What is cosign and why is image signing important?
- How does the container DNS work (127.0.0.11)?

### 5.10 Hands-on Labs — Containers
- [ ] Build a container from scratch using namespaces, cgroups, and chroot (no Docker)
- [ ] Create optimized multi-stage Dockerfile reducing image from 1GB to <50MB
- [ ] Set up container networking manually using veth pairs and bridges
- [ ] Implement container security scanning pipeline with Trivy in CI
- [ ] Build multi-platform images (AMD64 + ARM64) with buildx
- [ ] Configure rootless Podman and compare with Docker
- [ ] Set up ECR with lifecycle policies, scanning, and cross-account access
- [ ] Debug container networking issues using nsenter and tcpdump
- [ ] Implement BuildKit cache mounts for faster builds
- [ ] Sign container images with cosign and verify in admission controller

---

## Section 6 — Kubernetes

### 6.1 Architecture Overview

#### Control Plane Components
- **kube-apiserver**: RESTful API server, all communication goes through it, authentication → authorization (RBAC) → admission control → etcd
- **etcd**: distributed key-value store, Raft consensus, stores all cluster state, single source of truth
- **kube-scheduler**: watches unscheduled pods, selects node via filtering → scoring → binding
- **kube-controller-manager**: runs controllers (Deployment, ReplicaSet, Node, Job, Endpoint, ServiceAccount)
- **cloud-controller-manager**: cloud-specific controllers (Node, Route, LoadBalancer)

#### Node Components
- **kubelet**: pod lifecycle management, CRI calls to container runtime, health monitoring, status reporting
- **kube-proxy**: service networking, maintains iptables/IPVS rules for ClusterIP/NodePort/LoadBalancer
- **Container runtime**: containerd (default), CRI-O — implements CRI (Container Runtime Interface)

#### Communication Flows
- `kubectl` → API server (REST over HTTPS) → etcd
- Scheduler watches API server for unscheduled pods → updates pod spec with node binding
- Kubelet watches API server for pods assigned to its node → creates containers via CRI
- Controllers watch API server → reconcile desired vs actual state

### 6.2 etcd Deep Dive
- **Raft consensus**: leader election, log replication, term-based leadership
  - Leader handles all writes, replicates to followers, majority quorum (N/2+1)
  - Leader election on timeout (150-300ms default), heartbeats maintain leadership
- **Data model**: flat key-value, MVCC (multi-version), revisions for each key, watch on key ranges
- **Performance requirements**: <10ms write latency, SSD/NVMe storage, dedicated disk, low-latency network
- **Operations**: `etcdctl snapshot save/restore`, defragmentation (`etcdctl defrag`), compaction (auto or manual), member add/remove
- **Cluster sizing**: 3 nodes (tolerates 1 failure), 5 nodes (tolerates 2), never even numbers
- **Failure modes**: leader loss (re-election in seconds), split brain (prevented by quorum), disk full (read-only mode), slow disk (increased election timeouts, missed heartbeats)

### 6.3 Scheduler Deep Dive
- **Scheduling pipeline**: filter (predicates) → score (priorities) → bind
- **Filtering**: removes nodes that can't run the pod
  - NodeSelector/NodeAffinity: label-based node selection
  - Taints/Tolerations: nodes repel pods unless tolerated
  - Resource requirements: node must have enough allocatable CPU/memory
  - PodTopologySpread: maxSkew, topologyKey constraints
  - PodAffinity/AntiAffinity: co-locate or separate pods based on labels
  - Volume constraints: zone matching for PVs
- **Scoring**: ranks remaining nodes
  - LeastAllocated: prefer nodes with most available resources (spread)
  - MostAllocated: prefer nodes with least available resources (bin-pack)
  - NodeAffinity weight: higher score for preferred node affinity
  - PodTopologySpread: minimize skew score
- **Advanced**:
  - PriorityClasses: pod priority, preemption of lower-priority pods
  - Scheduling profiles: configure plugins per profile
  - Custom schedulers: deploy additional schedulers, `schedulerName` in pod spec
  - Descheduler: evict pods to rebalance (after node additions, policy changes)

### 6.4 Networking — CNI Deep Dive

#### Kubernetes Networking Model
- Every pod gets its own IP address
- Pods can communicate with all other pods without NAT
- Agents on a node can communicate with all pods on that node
- Flat network: no network-level isolation by default

#### CNI Plugins
- **AWS VPC CNI**:
  - Pods get IPs from VPC subnet (ENI secondary IPs)
  - Native VPC routing, no overlay overhead
  - Prefix delegation: /28 prefix per ENI slot = 16 IPs per slot, dramatically increases pod density
  - Tuning: WARM_ENI_TARGET, WARM_IP_TARGET, MINIMUM_IP_TARGET, WARM_PREFIX_TARGET
  - Custom networking: pods in different subnets than nodes (ENIConfig CRD)
  - Limitation: IP address exhaustion if VPC CIDR is small → use secondary CIDR (100.64.0.0/16)
- **Calico**: BGP-based routing (no overlay) or VXLAN/IP-in-IP overlay, rich network policies, eBPF dataplane option
- **Cilium**: eBPF-based, kube-proxy replacement, L7 network policies, Hubble observability, service mesh (sidecarless), transparent encryption

#### Service Networking
- **ClusterIP**: virtual IP, internal only, iptables/IPVS DNAT to pod endpoints
- **NodePort**: exposes service on each node's IP at static port (30000-32767)
- **LoadBalancer**: provisions cloud load balancer, routes to NodePort
- **ExternalName**: CNAME alias to external DNS name
- **Headless service** (`clusterIP: None`): no virtual IP, DNS returns pod IPs directly, used with StatefulSets

#### kube-proxy Modes
- **iptables** (default): O(n) chain rules, probability-based load balancing, conntrack for return traffic
- **IPVS**: O(1) hash lookup, multiple LB algorithms (rr, lc, wrr), better performance at scale
- **eBPF** (Cilium): replaces kube-proxy entirely, socket-level load balancing, lowest latency

#### CoreDNS
- Serves `cluster.local` domain: `<svc>.<ns>.svc.cluster.local`
- Pod DNS: `<pod-ip-dashed>.<ns>.pod.cluster.local`
- Headless services: return A records for all pod IPs
- Configuration: Corefile (ConfigMap), plugins (forward, cache, log, errors, health, ready)
- `ndots:5` issue: unqualified names get 5 search domain suffixes appended, causing 5× DNS queries → fix with FQDN (trailing dot) or reduce ndots

#### Network Policies
- **Default**: all traffic allowed if no NetworkPolicy exists
- **Default deny**: `podSelector: {}` with empty ingress/egress = deny all
- **Rules**: ingress (who can talk TO these pods), egress (where these pods can talk TO)
- **Selectors**: podSelector, namespaceSelector, ipBlock (CIDR ranges)
- **Requires CNI support**: Calico, Cilium, Antrea — VPC CNI alone does not support network policies

#### Ingress & Gateway API
- **Ingress**: path-based routing, TLS termination, host-based routing — Ingress controller required (nginx, ALB, Traefik)
- **Gateway API**: next-gen, role-oriented (infra provider, cluster admin, app developer), HTTPRoute, GRPCRoute, TLSRoute, more expressive than Ingress

### 6.5 Storage
- **Volume types**: emptyDir (ephemeral), hostPath (node disk), configMap, secret, projected, downwardAPI
- **PersistentVolume (PV)**: cluster-level storage resource, provisioned statically or dynamically
- **PersistentVolumeClaim (PVC)**: user's request for storage, binds to PV
- **StorageClass**: defines provisioner + parameters, reclaimPolicy (Delete/Retain), volumeBindingMode (Immediate/WaitForFirstConsumer)
- **CSI drivers**: EBS CSI (block), EFS CSI (NFS), FSx CSI (Lustre) — installed as Kubernetes DaemonSet + controller
- **Access modes**: ReadWriteOnce (RWO, single node), ReadOnlyMany (ROX, multi-node read), ReadWriteMany (RWX, multi-node write — EFS)
- **Volume expansion**: resize PVC (StorageClass must allow), filesystem resize on next mount
- **Snapshots**: VolumeSnapshot, VolumeSnapshotClass, VolumeSnapshotContent — backup and clone
- **StatefulSets**: volumeClaimTemplates — unique PVC per pod, stable storage across pod restarts

### 6.6 Workload Resources

#### Deployments
- **Rolling updates**: maxSurge (extra pods during update), maxUnavailable (pods that can be down), revision history
- **Rollback**: `kubectl rollout undo`, revision-based, automatic on failed deployment
- **Strategy**: RollingUpdate (default), Recreate (all down, all up)

#### StatefulSets
- **Ordered**: pods created/deleted in order (0, 1, 2...)
- **Stable identity**: `<statefulset>-<ordinal>`, predictable DNS names via headless service
- **Stable storage**: volumeClaimTemplates, PVC persists across pod restarts
- **Use cases**: databases, Kafka, ZooKeeper, etcd — anything needing stable identity and storage
- **Update strategies**: RollingUpdate (with partition for canary), OnDelete (manual)

#### DaemonSets
- **One pod per node**: logging agents, monitoring agents, CNI, kube-proxy
- **Update strategies**: RollingUpdate (maxUnavailable), OnDelete
- **Tolerations**: tolerate control-plane taints if needed (e.g., monitoring all nodes)

#### Jobs & CronJobs
- **Jobs**: run to completion, `completions`, `parallelism`, `backoffLimit`, `activeDeadlineSeconds`
- **CronJobs**: scheduled Jobs, cron syntax, `concurrencyPolicy` (Allow/Forbid/Replace), `startingDeadlineSeconds`

#### Autoscaling
- **HPA (Horizontal Pod Autoscaler)**:
  - Metrics: CPU, memory (resource metrics), custom metrics (Prometheus), external metrics (SQS queue depth)
  - Algorithm: `desiredReplicas = ceil(currentReplicas * (currentMetric / desiredMetric))`
  - Behavior: scaleUp/scaleDown policies, stabilization window, rate limiting
  - Requires: metrics-server or Prometheus adapter
- **VPA (Vertical Pod Autoscaler)**:
  - Modes: Off (recommendations only), Auto (evict and recreate with new resources), Initial (set on creation only)
  - Limitations: can't run with HPA on same CPU/memory metrics, causes pod restarts
- **KEDA**: event-driven autoscaling, scales to/from zero, 50+ scalers (SQS, Kafka, Prometheus, cron, HTTP)

### 6.7 Configuration & Secrets
- **ConfigMaps**: key-value config data, mount as volume or inject as env vars, not encrypted
- **Secrets**: base64-encoded (not encrypted by default), types: Opaque, kubernetes.io/tls, kubernetes.io/dockerconfigjson
- **Encryption at rest**: EncryptionConfiguration, KMS provider (AWS KMS envelope encryption)
- **External Secrets Operator**: syncs secrets from AWS Secrets Manager/Vault/GCP SM → Kubernetes Secrets
- **Sealed Secrets**: encrypt secrets client-side (kubeseal), store encrypted in Git, controller decrypts in cluster

### 6.8 Security

#### RBAC
- **Role/ClusterRole**: define permissions (verbs on resources)
- **RoleBinding/ClusterRoleBinding**: bind roles to subjects (users, groups, service accounts)
- **Least privilege**: avoid cluster-admin, scope roles to namespaces, use aggregated roles
- **Audit**: `kubectl auth can-i`, `kubectl-who-can`, review bindings regularly

#### Pod Security
- **Pod Security Standards**: Privileged (unrestricted), Baseline (minimally restrictive), Restricted (hardened)
- **Pod Security Admission**: enforce/audit/warn modes at namespace level via labels
- **Key restrictions (Restricted)**: non-root, drop ALL capabilities, read-only root filesystem, no hostPath, no privileged, seccomp profile required

#### Runtime Security
- **Falco**: eBPF/kernel module syscall monitoring, detect anomalies (shell in container, sensitive file access)
- **OPA/Gatekeeper**: admission controller, Rego policies, constraint templates
- **Kyverno**: Kubernetes-native policies, validate/mutate/generate, YAML-based (no Rego)

### 6.9 Cluster Operations
- **Upgrades**: control plane → add-ons → node groups (rolling), respect PDBs, test in staging first
- **Node management**: `kubectl cordon` (mark unschedulable), `kubectl drain` (evict pods + cordon), `kubectl uncordon`
- **Resource management**:
  - Requests: guaranteed resources, used for scheduling
  - Limits: maximum resources, OOM kill if memory exceeded, CPU throttled if exceeded
  - QoS classes: Guaranteed (requests=limits), Burstable (requests<limits), BestEffort (no requests/limits)
  - LimitRange: default requests/limits per namespace
  - ResourceQuota: total resource caps per namespace
- **Multi-tenancy**: namespaces, RBAC, network policies, resource quotas, Pod Security Standards

### 6.10 Advanced Patterns
- **Operator pattern**: Custom Resources (CRD) + custom controller = domain-specific automation
  - kubebuilder/operator-sdk for building operators
  - Examples: Prometheus Operator, cert-manager, CloudNativePG
- **Admission webhooks**: mutating (inject sidecars, defaults) and validating (policy enforcement)
- **Pod Disruption Budgets (PDB)**: minAvailable or maxUnavailable, protects during voluntary disruptions (drains, upgrades)
- **Finalizers**: prevent resource deletion until cleanup complete, stuck finalizers block deletion
- **Ephemeral containers**: `kubectl debug` — inject debug container into running pod, no restart needed

### 6.11 Failure Scenarios & Debugging — Kubernetes
- **Pending**: `kubectl describe pod` → Events section; causes: insufficient resources, unschedulable (taints, nodeSelector, affinity), PVC pending
- **CrashLoopBackOff**: `kubectl logs --previous`; causes: app crash, missing config/secret, OOM, liveness probe too aggressive, wrong command
- **ImagePullBackOff**: wrong image name/tag, registry auth (imagePullSecrets), rate limiting (Docker Hub), private registry access
- **Terminating stuck**: check finalizers (`kubectl get pod -o json | jq '.metadata.finalizers'`), PDB blocking, stuck preStop hook → remove finalizer or force delete
- **Service not reachable**: check selector matches pod labels, check endpoints (`kubectl get ep`), check network policy, verify kube-proxy running
- **Node NotReady**: SSH to node, check kubelet (`journalctl -u kubelet`), check disk/memory pressure, check container runtime
- **DNS failure**: check CoreDNS pods, test from pod (`nslookup kubernetes.default`), check ndots, check Corefile ConfigMap
- **HPA not scaling**: check metrics-server, `kubectl get hpa` for current metrics, verify metric name/threshold, check min/max replicas
- **etcd issues**: high latency → check disk (IOPS), defragment, compact; quota exceeded → increase quota, compact + defrag
- **OOMKilled**: check container memory limit vs actual usage, `kubectl top pod`, describe pod for Last State, increase limit or fix leak

### 6.12 Interview Questions — Kubernetes
- Explain Kubernetes architecture. What does each control plane component do?
- Walk through what happens when you run `kubectl apply -f deployment.yaml`.
- How does the scheduler decide where to place a pod?
- Explain Kubernetes service discovery and DNS.
- Deployment vs StatefulSet — when use each?
- How do network policies work? Design one for a 3-tier app.
- Explain RBAC. Design RBAC for a multi-team cluster.
- How does HPA work internally? What metrics can it use?
- Explain requests vs limits. What happens when exceeded?
- Debug a pod in CrashLoopBackOff — walk through your approach.
- Explain the Kubernetes networking model and pod-to-pod communication.
- How does AWS VPC CNI work? What is prefix delegation?
- What is an Operator? When would you build one?
- How do you handle secrets securely in Kubernetes?
- Explain pod topology spread constraints.
- How do you implement zero-downtime deployments?
- What is a PDB and why is it important?
- How does etcd store data and what is Raft consensus?
- Design a multi-tenant Kubernetes cluster.
- Ingress vs Gateway API — differences?
- How does KEDA differ from HPA?
- Explain CSI and how persistent volumes work.

### 6.13 Hands-on Labs — Kubernetes
- [ ] Deploy multi-tier app with Deployments, Services, Ingress, and Network Policies
- [ ] Set up HPA with custom Prometheus metrics
- [ ] Implement RBAC for multi-team access with namespace isolation
- [ ] Configure default-deny network policies with explicit allow rules
- [ ] Set up Velero backup/restore for cluster DR
- [ ] Build a simple Kubernetes operator with kubebuilder
- [ ] Debug 10+ broken deployments (Pending, CrashLoopBackOff, ImagePullBackOff, OOMKilled)
- [ ] Implement pod topology spread across AZs
- [ ] Set up External Secrets Operator with AWS Secrets Manager
- [ ] Configure Karpenter with spot instances and consolidation
- [ ] Deploy StatefulSet with PVCs and test pod failover
- [ ] Set up Cilium CNI with Hubble observability and kube-proxy replacement

---

## Section 7 — Terraform

### 7.1 IaC Fundamentals
- **What is IaC**: infrastructure defined in code, version-controlled, repeatable, auditable
- **Declarative vs Imperative**: Terraform (desired state) vs scripts (step-by-step)
- **Idempotency**: applying same config multiple times yields same result
- **Terraform vs alternatives**: Pulumi (real programming languages), CloudFormation (AWS-native), CDK (imperative → CFN), Crossplane (K8s-native)

### 7.2 Core Concepts

#### Resources & Data Sources
- **Resources**: infrastructure objects to create/manage (`resource "aws_instance" "web" {}`)
- **Data sources**: query existing infrastructure (`data "aws_ami" "latest" {}`)
- **Meta-arguments**: `count`, `for_each`, `depends_on`, `lifecycle`, `provider`

#### Variables & Outputs
- **Input variables**: `variable` block, types (string, number, bool, list, map, object, tuple), validation rules, sensitive
- **Outputs**: expose values for other modules or CLI, `output` block, sensitive output
- **Locals**: computed intermediate values, reduce repetition (`locals {}`)
- **Variable precedence**: env vars → tfvars → -var flag → default

#### Expressions & Functions
- **Conditionals**: `condition ? true_val : false_val`
- **For expressions**: `[for s in var.list : upper(s)]`, filtering with `if`
- **Splat**: `aws_instance.web[*].id`
- **Dynamic blocks**: generate repeated nested blocks with `for_each`
- **Functions**: `join()`, `split()`, `lookup()`, `merge()`, `flatten()`, `coalesce()`, `try()`, `templatefile()`, `cidrsubnet()`

### 7.3 State Management Deep Dive

#### State Fundamentals
- **Purpose**: maps real resources to config, tracks metadata, stores resource attributes, dependency graph
- **State file**: `terraform.tfstate` — JSON, contains all resource state and metadata
- **State locking**: prevent concurrent modifications, DynamoDB for S3 backend

#### Remote Backend
- **S3 + DynamoDB**: standard AWS backend, S3 for state, DynamoDB for locking
  ```
  backend "s3" {
    bucket         = "terraform-state"
    key            = "env/prod/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
  ```
- **State per environment**: separate state files (different S3 keys or backends)
- **Cross-state references**: `terraform_remote_state` data source or outputs

#### State Operations
- `terraform state list` — list all resources in state
- `terraform state show <resource>` — show resource details
- `terraform state mv` — rename/move resources without destroy/recreate
- `terraform state rm` — remove resource from state (resource remains in cloud)
- `terraform import` — import existing resources into state
- `terraform force-unlock <lock-id>` — release stuck lock

#### State Security
- State contains sensitive data (passwords, keys) in plaintext
- Encrypt at rest (S3 SSE), restrict access (IAM policies), never commit to Git
- Use `sensitive = true` on variables and outputs to redact from CLI output

### 7.4 Modules

#### Module Structure
```
modules/vpc/
├── main.tf        # resource definitions
├── variables.tf   # input variables
├── outputs.tf     # output values
├── versions.tf    # required providers and versions
└── README.md      # documentation
```

#### Module Best Practices
- **Single responsibility**: one module = one logical unit (VPC, EKS, IAM role)
- **Clear interfaces**: well-defined inputs/outputs, documented variables
- **Version pinning**: `source = "git::https://...?ref=v1.2.3"` or registry with version constraint
- **Composition over inheritance**: compose small modules into larger configurations
- **Testing**: terratest (Go), `terraform test` (native, v1.6+), `terraform validate`, `terraform plan`

#### Module Sources
- Local: `source = "./modules/vpc"`
- Terraform Registry: `source = "terraform-aws-modules/vpc/aws"` with `version = "~> 5.0"`
- Git: `source = "git::https://github.com/org/module.git?ref=v1.0.0"`
- S3: `source = "s3::https://bucket.s3.amazonaws.com/module.zip"`

### 7.5 Terraform Internals
- **Dependency graph**: DAG (Directed Acyclic Graph), automatic dependency resolution, `depends_on` for implicit dependencies
- **Parallelism**: default 10 concurrent operations (`-parallelism=N`)
- **Plan execution**: refresh (read current state) → diff (compare desired vs actual) → create plan → apply changes
- **Provider plugin protocol**: gRPC, downloaded during `terraform init`, cached in `.terraform/providers/`
- **Lock file**: `.terraform.lock.hcl` — pins provider versions and hashes, commit to Git

#### Resource Lifecycle
- `create_before_destroy`: create replacement before destroying original (e.g., ASG, DNS)
- `prevent_destroy`: prevent accidental destruction (production databases)
- `ignore_changes`: ignore specific attribute changes (tags managed externally, auto-scaling desired count)
- `replace_triggered_by`: force replacement when referenced resource changes

#### Import
- `terraform import aws_instance.web i-1234567890` — import existing resource
- Import blocks (v1.5+): `import { to = aws_instance.web; id = "i-123" }` — declarative import
- Generate config: `terraform plan -generate-config-out=generated.tf` — auto-generate config for imported resources

### 7.6 Advanced Patterns

#### Workspaces
- CLI workspaces: `terraform workspace new dev` — same config, different state files
- Use `terraform.workspace` in config for environment-specific values
- Limitation: shares backend config, not suitable for drastically different environments
- Alternative: directory-based environments (recommended for production)

#### Terragrunt
- DRY configurations: shared `terragrunt.hcl` with `include` blocks
- Remote state management: automatic backend configuration
- Dependency orchestration: `dependency` blocks, `mock_outputs` for plan
- `run_all`: apply/plan across multiple modules
- When to use: multi-account, multi-environment, many modules with shared config

#### Policy as Code
- **Checkov**: static analysis, CIS benchmarks, custom policies in Python/YAML
- **tfsec**: security-focused scanning, severity levels, custom rules
- **OPA/Conftest**: Rego policies on `terraform plan` JSON output
- **Sentinel** (Terraform Cloud/Enterprise): policy-as-code framework, advisory/soft/hard enforcement

### 7.7 Production Patterns

#### Repository Structure
```
infrastructure/
├── modules/             # reusable modules
│   ├── vpc/
│   ├── eks/
│   └── iam/
├── environments/
│   ├── dev/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── terraform.tfvars
│   ├── staging/
│   └── prod/
├── global/              # shared resources (IAM, Route53)
└── terragrunt.hcl       # (if using Terragrunt)
```

#### CI/CD Integration
- **Plan on PR**: `terraform plan` output as PR comment, reviewers see exact changes
- **Apply on merge**: automated apply after PR merge to main/environment branch
- **Drift detection**: scheduled `terraform plan` to detect out-of-band changes
- **Approval gates**: manual approval for production applies
- **State locking**: CI ensures single concurrent apply per state file

#### Terraform for EKS (Goodnotes-Relevant)
- **Community modules**: `terraform-aws-modules/vpc/aws`, `terraform-aws-modules/eks/aws`
- **EKS module**: managed node groups, Fargate profiles, add-ons, IRSA, Karpenter setup
- **Helm provider**: deploy Helm charts (ArgoCD, Datadog, cert-manager) via Terraform
- **Kubernetes provider**: manage K8s resources (namespaces, RBAC) — use sparingly, prefer GitOps

### 7.8 Failure Scenarios & Debugging — Terraform
- **State lock stuck**: `terraform force-unlock <ID>`, check DynamoDB for stale locks
- **State corruption**: restore from S3 versioned backup, manual state surgery with `terraform state` commands
- **Dependency cycle**: restructure resources, use `depends_on`, split into separate modules
- **Provider version conflict**: pin versions in `required_providers`, use `.terraform.lock.hcl`
- **Unexpected destroy/recreate**: check `lifecycle` rules, `ignore_changes`, ForceNew attributes (`name` change on some resources)
- **Import failures**: resource already in state, state/resource mismatch, unsupported resource type
- **Apply timeout**: increase `timeouts` block, check AWS API rate limits, retry
- **Drift**: `terraform plan` shows unexpected changes, someone modified resource outside Terraform → import or accept

### 7.9 Interview Questions — Terraform
- Explain the Terraform workflow (init, plan, apply).
- How does Terraform state work and why is it important?
- What happens if two people run `terraform apply` simultaneously?
- How do you manage state for multiple environments?
- Explain the dependency graph and how Terraform determines order.
- `count` vs `for_each` — when use each?
- How do you handle secrets in Terraform?
- Explain `create_before_destroy` and when you'd use it.
- How would you refactor a large Terraform codebase?
- What is Terragrunt and when would you use it?
- How do you test Terraform code?
- Workspaces vs directory-based environments — tradeoffs?
- How do you handle state drift in production?
- Design a Terraform CI/CD pipeline with plan-on-PR and apply-on-merge.
- How would you structure Terraform for a multi-account AWS organization?
- How do you import existing infrastructure into Terraform?

### 7.10 Hands-on Labs — Terraform
- [ ] Set up remote backend with S3 + DynamoDB locking
- [ ] Build reusable VPC module with configurable CIDRs, subnets, and NAT
- [ ] Deploy EKS cluster using community modules
- [ ] Implement CI/CD pipeline with plan-on-PR (GitHub Actions)
- [ ] Set up multi-environment management (dev/staging/prod)
- [ ] Implement policy-as-code scanning with Checkov in CI
- [ ] Import existing AWS resources into Terraform state
- [ ] Refactor monolithic config into modules with state migration
- [ ] Set up Terragrunt for DRY multi-account infrastructure
- [ ] Implement automated drift detection with scheduled plans

---

## Section 8 — CI/CD

### 8.1 CI/CD Fundamentals
- **Continuous Integration**: build and test on every commit, fast feedback, catch issues early
- **Continuous Delivery**: every commit is deployable, manual release gate
- **Continuous Deployment**: every commit auto-deploys to production, requires mature testing/monitoring
- **Pipeline stages**: build → unit test → lint → security scan → integration test → artifact publish → deploy
- **Artifact management**: container images (ECR), versioning (git SHA, semver), immutable tags

### 8.2 GitHub Actions Deep Dive

#### Architecture
- **Events**: triggers (push, pull_request, workflow_dispatch, schedule, repository_dispatch)
- **Workflows**: YAML files in `.github/workflows/`, triggered by events
- **Jobs**: run on a runner, contain steps, can run in parallel or depend on each other (`needs`)
- **Steps**: individual tasks — `uses` (actions) or `run` (shell commands)
- **Actions**: reusable units (`actions/checkout`, `aws-actions/configure-aws-credentials`)

#### Runners
- **GitHub-hosted**: Ubuntu, macOS, Windows — clean environment each run, limited resources
- **Self-hosted**: your infrastructure, custom software, persistent environment
- **ARC (Actions Runner Controller)**: self-hosted runners on Kubernetes, autoscaling based on workflow demand, ephemeral runners (fresh pod per job)
- **Runner groups**: organize runners, restrict access to specific repos/orgs

#### Key Features
- **Matrix strategy**: test across multiple versions/platforms `matrix: { node: [16, 18, 20], os: [ubuntu, macos] }`
- **Concurrency groups**: prevent concurrent runs `concurrency: { group: deploy-prod, cancel-in-progress: false }`
- **Environments**: protection rules, required reviewers, wait timers, deployment branches
- **Caching**: `actions/cache` — dependency cache (npm, pip, go), Docker layer cache, restore keys
- **Artifacts**: `actions/upload-artifact` / `download-artifact` — share data between jobs
- **Reusable workflows**: `workflow_call` trigger, inputs/outputs/secrets, shared across repos
- **Composite actions**: bundle multiple steps into one action, `action.yml`, custom inputs/outputs

#### OIDC Federation (AWS)
- GitHub Actions assumes AWS role without long-lived credentials
- Flow: GitHub OIDC token → AWS STS AssumeRoleWithWebIdentity → temporary credentials
- Trust policy conditions: `token.actions.githubusercontent.com:sub` matches repo/branch/environment
- Eliminates: AWS access keys stored as secrets, credential rotation

#### Security
- **Permissions**: `permissions` block — least-privilege GITHUB_TOKEN scope
- **Secret management**: repository/environment/org secrets, never logged
- **Dependabot**: automated dependency updates, security alerts
- **CodeQL**: SAST scanning, supports multiple languages
- **Branch protection**: required checks, required reviews, signed commits

### 8.3 Pipeline Optimization
- **Parallelization**: independent jobs run simultaneously, fan-out/fan-in pattern
- **Caching strategies**:
  - Dependency cache: `actions/cache` with lock file hash key
  - Docker layer cache: `docker/build-push-action` with cache-from/cache-to
  - Build cache: remote build cache (BuildKit), incremental compilation
- **Conditional execution**: path filters (`paths:`, `paths-ignore:`), `if: contains(github.event.head_commit.message, '[skip ci]')`
- **Test optimization**: test splitting, parallel test execution, test impact analysis (run only affected tests)

### 8.4 Advanced CI/CD Patterns
- **GitOps deployment flow**: CI builds image → pushes to ECR → updates Helm values/kustomize in GitOps repo → ArgoCD syncs
- **Multi-environment**: dev (auto-deploy on push) → staging (auto-deploy on merge) → prod (manual approval + canary)
- **Infrastructure CI**: Terraform plan on PR, apply on merge, drift detection cron
- **Monorepo CI**: path-based filtering, detect changed services, build only affected
- **Deployment strategies**: rolling update, blue-green, canary, A/B testing — via ArgoCD/Argo Rollouts

### 8.5 Security in CI/CD
- **Supply chain security**: SLSA framework, provenance attestation, artifact signing (cosign)
- **Secrets**: never in code, OIDC for cloud auth, short-lived credentials, secrets rotation
- **Dependency scanning**: Dependabot, Snyk, `npm audit`, `pip-audit`, `govulncheck`
- **Container scanning**: Trivy, Grype, ECR native scanning — fail build on critical CVEs
- **SAST/DAST**: CodeQL, Semgrep, SonarQube — static analysis in PR

### 8.6 Failure Scenarios & Debugging — CI/CD
- **Flaky tests**: test isolation, retry configuration, track flaky test rate
- **Cache miss**: check cache key design, fallback keys (`restore-keys`), cache expiry
- **Runner disk full**: cleanup steps, ephemeral runners, prune Docker cache
- **OIDC not working**: check trust policy, verify subject claim format, audience configuration
- **Rate limiting**: Docker Hub (100 pulls/6h unauthenticated), GitHub API limits, ECR throttling
- **Pipeline timeout**: step/job timeouts, deadlock detection, resource exhaustion

### 8.7 Interview Questions — CI/CD
- Design a CI/CD pipeline for microservices deployed to Kubernetes.
- How does GitHub Actions OIDC federation work with AWS?
- How would you optimize a 30-minute CI pipeline?
- Reusable workflows vs composite actions — differences and when to use each?
- How do you handle secrets in CI/CD securely?
- What is SLSA and why does it matter?
- How do you implement progressive delivery in CI/CD?
- Explain the GitOps deployment workflow.
- How would you set up CI/CD for a monorepo?
- What are concurrency groups and when to use them?
- How do you handle database migrations in CI/CD?
- Design a rollback strategy for failed deployments.
- How do you test Terraform changes in CI?
- Explain self-hosted runners on Kubernetes with ARC.
- How do you handle multi-environment deployments?

### 8.8 Hands-on Labs — CI/CD
- [ ] Build complete GitHub Actions pipeline: build, test, scan, push to ECR, deploy
- [ ] Set up OIDC federation between GitHub Actions and AWS
- [ ] Implement Docker layer caching and dependency caching
- [ ] Create reusable workflows and composite actions
- [ ] Set up self-hosted runners on EKS with ARC
- [ ] Implement matrix builds for multi-platform Docker images
- [ ] Set up Terraform CI/CD with plan-on-PR and apply-on-merge
- [ ] Implement security scanning pipeline (Trivy + CodeQL + Dependabot)
- [ ] Build monorepo CI with path-based filtering
- [ ] Set up deployment pipeline with staging → manual approval → production

---

## Section 9 — GitOps

### 9.1 GitOps Fundamentals
- **Principles**: Git as single source of truth, declarative desired state, automated reconciliation, continuous observation
- **Push vs Pull**: CI push (`kubectl apply`) vs GitOps pull (ArgoCD/Flux watches Git, syncs to cluster)
- **Benefits**: audit trail (Git history), rollback (git revert), drift detection, self-healing, consistency

### 9.2 ArgoCD Deep Dive

#### Architecture
- **API Server**: REST/gRPC API, Web UI, SSO integration, RBAC
- **Repo Server**: clones Git repos, generates manifests (Helm, Kustomize, plain YAML)
- **Application Controller**: watches applications, compares desired (Git) vs live (cluster), reconciles
- **Redis**: caching layer for controller and API server
- **Dex**: OIDC/SAML SSO provider (optional)

#### Application CRD
- **Source**: Git repo (URL, path, targetRevision), Helm chart (repo, chart, version), Kustomize
- **Destination**: cluster (in-cluster or remote), namespace
- **Sync policy**: manual or auto-sync, self-heal (revert manual changes), prune (delete removed resources)

#### Sync Mechanics
- **Sync strategies**: manual sync, auto-sync (on Git change), self-heal (on drift), prune orphaned resources
- **Sync phases**: PreSync → Sync → PostSync → SyncFail hooks
- **Sync waves**: annotation `argocd.argoproj.io/sync-wave: "1"` — order resource creation (e.g., namespace before deployment)
- **Resource hooks**: PreSync Jobs (database migrations), PostSync (notifications, smoke tests)
- **Health assessment**: built-in checks (Deployment, StatefulSet, Service), custom Lua health checks

#### Multi-Cluster & Scaling
- **ApplicationSets**: template applications across clusters/environments
  - Generators: list (static), cluster (from ArgoCD cluster secrets), git (directory/file), matrix (combine), merge
  - Progressive sync: roll out across clusters sequentially
- **App of Apps**: bootstrap pattern — one ArgoCD Application manages other Applications
- **Cluster secrets**: ArgoCD stores cluster credentials as Kubernetes secrets
- **Scaling**: HA deployment (3+ replicas), sharding application controller, repo server caching

#### Secrets in GitOps
- **Problem**: secrets can't be stored in Git in plaintext
- **Solutions**:
  - External Secrets Operator: ExternalSecret CRD → syncs from AWS Secrets Manager/Vault → K8s Secret
  - Sealed Secrets: encrypt with kubeseal → store encrypted in Git → controller decrypts in cluster
  - SOPS: Mozilla SOPS encrypts values in YAML/JSON, decrypt with KMS key
  - Vault: CSI provider or sidecar injector

### 9.3 Argo Rollouts — Progressive Delivery

#### Canary Deployments
- Step-based: `setWeight: 20` → `pause: { duration: 5m }` → analysis → `setWeight: 50` → `promote`
- Traffic management: Istio VirtualService, Nginx annotations, ALB target group weights
- Analysis templates: automated checks during canary (success rate, latency, error rate)

#### Blue-Green Deployments
- Active service + preview service
- `autoPromotionEnabled`: auto-promote after preview is healthy
- `scaleDownDelaySeconds`: keep old version running for quick rollback
- `previewReplicaCount`: scale preview separately

#### Analysis Templates
- **Prometheus queries**: `successRate = sum(rate(http_requests_total{status=~"2.."}[5m])) / sum(rate(http_requests_total[5m]))`
- **Datadog queries**: `avg:http.request.duration{service:myapp}.as_rate() < 0.5`
- **Web analysis**: HTTP endpoint returning pass/fail JSON
- **Job analysis**: run a Kubernetes Job, success = pass
- **Failure action**: automatic rollback on failed analysis

### 9.4 Helm Deep Dive

#### Chart Structure
```
mychart/
├── Chart.yaml        # chart metadata, version, dependencies
├── values.yaml       # default values
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── _helpers.tpl   # named templates/partials
│   └── NOTES.txt      # post-install notes
├── charts/           # dependency charts
└── tests/            # helm test pods
```

#### Templating
- **Go templates**: `{{ .Values.replicaCount }}`, `{{ .Release.Name }}`, `{{ .Chart.Name }}`
- **Flow control**: `if/else`, `range`, `with`
- **Named templates**: `{{ define "mychart.labels" }}` in `_helpers.tpl`, `{{ include "mychart.labels" . | nindent 4 }}`
- **Functions**: `default`, `quote`, `toYaml`, `nindent`, `tpl`, `required`, `lookup`
- **Value precedence**: defaults (values.yaml) < parent chart values < `-f custom.yaml` < `--set key=val`

#### Production Patterns
- **Environment values**: `values-dev.yaml`, `values-staging.yaml`, `values-prod.yaml`
- **Library charts**: shared templates, no rendered resources, `type: library` in Chart.yaml
- **Umbrella charts**: parent chart with multiple sub-charts as dependencies
- **OCI registries**: store charts in ECR/GHCR, `helm push/pull` with OCI URLs
- **Testing**: `helm lint`, `helm template`, `helm test`, `ct lint-and-install` (chart-testing)

### 9.5 Kustomize
- **Base + Overlays**: base directory with common resources, overlay directories for environment-specific patches
- **Patches**: strategic merge patch, JSON patch, inline patches
- **Generators**: `configMapGenerator`, `secretGenerator` — auto-append hash suffix for rollout on change
- **Transformers**: `namePrefix`, `nameSuffix`, `commonLabels`, `commonAnnotations`, `images` (override image tags)
- **Components**: reusable Kustomize fragments, `kind: Component`

### 9.6 Failure Scenarios & Debugging — GitOps
- **OutOfSync won't sync**: resource hooks failing, admission webhook rejecting, PDB blocking, resource already exists (owned by another app)
- **Sync stuck**: PreSync Job failing, resource ordering issue (use sync waves)
- **Drift detected**: manual `kubectl` changes, self-heal not enabled → enable self-heal or revert manual changes
- **Application degraded**: health check failing, underlying pods unhealthy → check pod events/logs
- **Repo server overload**: too many apps, large repos → increase repo server replicas, caching, manifest generation timeout
- **Canary stuck**: analysis template failing, flapping metrics, incorrect thresholds → check analysis run, adjust queries
- **Helm render failure**: invalid values, type mismatch, missing required values → `helm template` locally to debug

### 9.7 Interview Questions — GitOps
- What is GitOps and how does it differ from traditional CI/CD?
- Explain ArgoCD architecture and reconciliation loop.
- How does ArgoCD detect drift?
- What is App of Apps pattern and when to use it?
- How do you manage secrets in GitOps?
- Explain ApplicationSets and their generators.
- How do you handle database migrations in GitOps?
- Auto-sync vs self-heal — difference?
- How do you implement canary deployments with Argo Rollouts?
- Explain sync waves and hooks in ArgoCD.
- How do you manage multiple environments with ArgoCD?
- Helm vs Kustomize — when use each?
- How do you handle Helm chart dependencies?
- Explain analysis templates in Argo Rollouts.
- How would you set up ArgoCD for multi-cluster deployment?

### 9.8 Hands-on Labs — GitOps
- [ ] Deploy ArgoCD and set up App of Apps pattern
- [ ] Implement multi-environment deployment with Kustomize overlays
- [ ] Set up Argo Rollouts canary with Prometheus/Datadog analysis
- [ ] Configure ApplicationSets for multi-cluster deployment
- [ ] Implement External Secrets Operator with ArgoCD
- [ ] Set up ArgoCD Slack notifications
- [ ] Build production Helm chart with environment-specific values
- [ ] Implement blue-green deployment with Argo Rollouts

---

## Section 10 — Monitoring and Observability

### 10.1 Observability Fundamentals
- **Three pillars**: metrics (numeric time-series), logs (text events), traces (request paths)
- **Monitoring vs Observability**: known unknowns (dashboards, alerts) vs unknown unknowns (explore, correlate, ask new questions)
- **Golden signals**: latency, traffic, errors, saturation
- **RED method** (services): Rate, Errors, Duration
- **USE method** (resources): Utilization, Saturation, Errors
- **SLIs, SLOs, SLAs**: measurable indicators → target objectives → contractual agreements
- **Cardinality**: number of unique label combinations — high cardinality = memory/cost explosion

### 10.2 Metrics

#### Metric Types
- **Counter**: monotonically increasing (requests_total, errors_total) — use `rate()` for per-second
- **Gauge**: goes up and down (temperature, queue_depth, memory_usage)
- **Histogram**: distribution of values in buckets (request_duration_seconds) — use `histogram_quantile()` for percentiles
- **Summary**: client-side percentile calculation — less flexible than histograms, avoid in new code

#### Prometheus
- **Architecture**: scrape-based (pull model), targets discovered via service discovery, local TSDB storage
- **PromQL**:
  - `rate(http_requests_total[5m])` — per-second rate over 5 minutes
  - `histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))` — p99 latency
  - `sum by (service) (rate(http_requests_total{status=~"5.."}[5m]))` — 5xx rate per service
  - `increase(errors_total[1h])` — total increase over 1 hour
  - Aggregation operators: `sum`, `avg`, `max`, `min`, `count`, `topk`, `bottomk`
  - Vector matching: `on()`, `ignoring()`, `group_left()`, `group_right()`
- **Recording rules**: pre-compute expensive queries, store as new time series
- **Alerting rules**: condition → pending → firing, `for` duration, labels, annotations
- **Prometheus Operator**: ServiceMonitor, PodMonitor CRDs for target discovery

#### Kubernetes Metrics Stack
- **metrics-server**: CPU/memory for HPA, `kubectl top`
- **kube-state-metrics**: Kubernetes object state (deployment replicas, pod status, node conditions)
- **node-exporter**: host-level metrics (CPU, memory, disk, network)
- **cAdvisor**: container-level resource usage (built into kubelet)

### 10.3 Logging

#### Structured Logging
- **JSON format**: machine-parseable, consistent fields (timestamp, level, message, service, traceID)
- **Correlation IDs**: trace ID, request ID — correlate logs across services
- **Log levels**: DEBUG (development), INFO (normal operations), WARN (unexpected but handled), ERROR (failures), FATAL (crash)

#### Log Aggregation in Kubernetes
- **DaemonSet pattern**: Fluent Bit/Fluentd DaemonSet reads container stdout/stderr from `/var/log/containers/`
- **Sidecar pattern**: per-pod log forwarder — more control, higher resource overhead
- **Pipeline**: collection (Fluent Bit) → processing (filters, parsing) → storage (S3, Elasticsearch, Loki) → querying

#### Log Platforms
- **ELK/EFK**: Elasticsearch (storage/search) + Logstash/Fluentd (processing) + Kibana (visualization)
- **Loki**: label-based log aggregation (like Prometheus for logs), LogQL, cost-effective (S3 storage), no full-text indexing
- **CloudWatch Logs**: AWS native, Log Insights query language, metric filters, alarms

### 10.4 Distributed Tracing
- **Concepts**: trace (end-to-end request), span (single operation), context propagation (trace ID passed between services)
- **OpenTelemetry**: vendor-neutral SDK + collector + exporters, auto-instrumentation, unified metrics/logs/traces
- **Sampling**: head-based (decide at start), tail-based (decide after trace completes), rate-limiting
- **Trace context**: W3C TraceContext standard, B3 propagation (Zipkin), `traceparent` header
- **Backends**: Jaeger, Tempo (Grafana), X-Ray (AWS), Datadog APM

### 10.5 Datadog Deep Dive (Goodnotes-Relevant)

#### Architecture
- **Datadog Agent**: DaemonSet on each node, collects metrics/logs/traces
- **Cluster Agent**: centralized metadata, external metrics for HPA, admission controller for APM injection
- **Datadog Operator**: DatadogAgent CRD for deployment configuration

#### Core Features
- **Infrastructure monitoring**: host maps, process monitoring, live containers, container maps
- **APM**: auto-instrumentation (Python, Go, Node, Java), service map, trace analytics, error tracking
- **Log management**: pipeline processing, parsing (grok), indexing, archives (S3), log-to-metric, live tail
- **Custom metrics**: DogStatsD protocol, `statsd.increment()`, `statsd.gauge()`, tagging strategy (`env`, `service`, `version`)
- **Synthetics**: API tests, browser tests, multi-step, CI/CD integration

#### Kubernetes Monitoring
- **Container Insights**: pod/container metrics, resource usage, orchestrator checks
- **Autodiscovery**: annotations on pods/services for custom integrations
- **Unified tagging**: `DD_ENV`, `DD_SERVICE`, `DD_VERSION` — correlate metrics, logs, traces
- **ADOT**: AWS Distro for OpenTelemetry, can export to Datadog

#### Dashboards & Alerting
- **Dashboards**: template variables (env, service, cluster), SLO widget, anomaly detection widget, heatmaps
- **Monitors**: metric, log, APM, composite (combine multiple conditions), forecast, anomaly, outlier
- **Notification channels**: Slack, PagerDuty, email, webhook
- **Downtime**: scheduled maintenance windows, suppress alerts

#### Cost Management
- Custom metric costs: per unique time series (tag combination), control cardinality
- Log indexing: retain high-value logs, archive rest to S3, exclusion filters
- Trace sampling: control ingestion rate, intelligent retention

### 10.6 Alerting Best Practices
- **Alert on symptoms, not causes**: alert on latency/errors (user impact), not CPU usage
- **Multi-window, multi-burn-rate**: fast burn (2% budget consumed in 1h) → page, slow burn (5% in 6h) → ticket
- **Reduce noise**: aggregate, deduplicate, set appropriate thresholds, avoid alert storms
- **Runbooks**: every alert links to a runbook with investigation steps
- **Escalation**: severity-based routing, auto-escalation after timeout
- **SLO-based alerting**: error budget burn rate alerts instead of static thresholds

### 10.7 Failure Scenarios — Observability
- **Metrics gap**: scrape target down, Prometheus OOM, network partition → check target health, increase resources
- **High cardinality**: metric explosion (too many label values) → relabeling to drop, aggregate, reduce labels
- **Missing logs**: Fluent Bit crash, buffer overflow, log rotation misconfigured → check DaemonSet pods, buffer config
- **Trace gaps**: sampling too aggressive, context propagation broken → verify sampling config, check instrumentation
- **Alert storm**: cascading failures, correlated alerts → alert grouping, inhibition rules, dependency-based suppression
- **Datadog agent high resource**: too many integrations, log volume → tune collection, exclude noisy namespaces

### 10.8 Interview Questions — Monitoring
- Explain the three pillars of observability.
- What are SLIs, SLOs, and SLAs? Give examples for a web API.
- How does Prometheus scraping work?
- Write a PromQL query for 99th percentile latency.
- How would you set up monitoring for a Kubernetes cluster?
- RED vs USE — when to use each?
- How do you handle high-cardinality metrics?
- Explain distributed tracing and context propagation.
- How do you design alerting to avoid alert fatigue?
- What is error budget and how do you alert on it?
- How does Datadog APM work in Kubernetes?
- Design an observability stack for microservices.
- How do you correlate metrics, logs, and traces?
- Explain log aggregation patterns in Kubernetes.
- How would you debug a latency spike using observability tools?

### 10.9 Hands-on Labs — Monitoring
- [ ] Deploy Prometheus + Grafana on Kubernetes with ServiceMonitors
- [ ] Set up Datadog agent on EKS with APM and log collection
- [ ] Create Datadog dashboards with SLO tracking
- [ ] Implement distributed tracing with OpenTelemetry
- [ ] Set up error budget burn rate alerting
- [ ] Configure Fluent Bit for log aggregation
- [ ] Write PromQL queries for RED metrics and SLI calculations
- [ ] Set up Datadog Synthetics for API health monitoring
- [ ] Implement custom metrics with DogStatsD
- [ ] Set up on-call rotation with PagerDuty integration

---

## Section 11 — Security

### 11.1 Security Fundamentals
- **Defense in depth**: multiple security layers — network, host, container, application, data
- **Principle of least privilege**: minimum access needed, just-in-time access, time-bounded
- **Zero trust**: never trust, always verify, micro-segmentation, identity-based access
- **CIA triad**: Confidentiality, Integrity, Availability
- **Threat modeling**: STRIDE (Spoofing, Tampering, Repudiation, Information Disclosure, DoS, Elevation of Privilege)

### 11.2 Container Security
- **Image security**: minimal base (distroless/alpine), vulnerability scanning (Trivy/Grype/Snyk), image signing (cosign), SBOM generation (syft)
- **Runtime security**: read-only rootfs, drop ALL caps + add specific, no-new-privileges, seccomp profiles, AppArmor
- **Non-root**: USER directive, `runAsNonRoot: true`, `runAsUser: 1000`, `fsGroup` for volume permissions
- **Supply chain**: SLSA framework (levels 1-4), provenance attestation, reproducible builds, in-toto
- **Registry security**: private registries, image pull secrets, admission controllers for image validation, tag immutability
- **Build security**: BuildKit secret mounts, multi-stage builds, no secrets in layers, `.dockerignore`

### 11.3 Kubernetes Security
- **RBAC**: least-privilege roles, audit with `kubectl auth can-i`, avoid `cluster-admin`, use namespaced roles
- **Pod Security Standards**: Privileged (unrestricted), Baseline (prevent known escalations), Restricted (hardened)
- **Pod Security Admission**: namespace labels for enforce/audit/warn modes
- **Network Policies**: default deny all, explicit allow, namespace isolation, egress control
- **Secrets**: encryption at rest (KMS provider), External Secrets Operator, avoid env vars (use volume mounts)
- **Service accounts**: disable auto-mount (`automountServiceAccountToken: false`), bound tokens, IRSA/Pod Identity
- **Admission control**: OPA/Gatekeeper (Rego policies), Kyverno (YAML policies) — enforce no privileged pods, required labels, resource limits
- **Runtime security**: Falco (syscall monitoring), detect anomalies (shell in container, unexpected network)
- **Audit logging**: audit policy (levels: None, Metadata, Request, RequestResponse), audit sinks

### 11.4 AWS Security
- **IAM**: no long-lived credentials, role-based access, permission boundaries, SCPs, regular access reviews
- **Network**: VPC isolation, SGs, NACLs, VPC Flow Logs, traffic mirroring, private endpoints
- **Encryption**: at rest (KMS for EBS, S3, RDS, EKS secrets), in transit (TLS everywhere, ACM certificates)
- **Secrets**: AWS Secrets Manager (rotation, RDS integration), Parameter Store (simple, cheaper)
- **Detection**: GuardDuty (threat detection), Security Hub (aggregated findings), Inspector (vulnerability scanning), Macie (S3 data classification)
- **Compliance**: AWS Config rules, conformance packs, automated remediation with EventBridge + Lambda
- **Incident response**: CloudTrail analysis, VPC Flow Logs forensics, automated containment

### 11.5 Secrets Management
- **Vault**: secret engines (KV, database, PKI, AWS), auth methods (Kubernetes, OIDC, AppRole), policies, agent sidecar injection
- **External Secrets Operator**: SecretStore (credentials), ExternalSecret (what to sync), supports AWS SM, Vault, GCP SM, Azure KV
- **Sealed Secrets**: kubeseal encrypts → store in Git → controller decrypts in cluster, key rotation
- **SOPS**: encrypts values in YAML/JSON, KMS/PGP keys, GitOps-compatible, `sops -e/-d`

### 11.6 Security Scanning & Compliance
- **Infrastructure scanning**: Checkov, tfsec, KICS, Prowler (AWS), kube-bench (CIS K8s benchmark)
- **Container scanning**: Trivy (comprehensive), Grype (fast), ECR native scanning
- **SAST**: CodeQL (GitHub), Semgrep (pattern matching), SonarQube
- **Policy as Code**: OPA/Rego, Kyverno policies, Conftest (test configs against policies)
- **Compliance**: CIS benchmarks (Kubernetes, AWS, Docker), SOC 2, HIPAA — automated checks

### 11.7 Interview Questions — Security
- How would you secure a Kubernetes cluster from scratch?
- Explain least privilege in AWS IAM — how do you implement it?
- How do you handle secrets in Kubernetes?
- What is mTLS and how to implement it?
- How would you respond to a security incident in Kubernetes?
- Explain Pod Security Standards and enforcement.
- How do you implement network segmentation in Kubernetes?
- Design RBAC for a multi-team cluster.
- How do you scan container images in CI/CD?
- Explain the SLSA framework.
- How do you manage TLS certificates in Kubernetes?
- What security controls for an EKS cluster?
- How does OPA/Gatekeeper work?
- Explain zero trust for microservices.
- How to detect and respond to a compromised container?

### 11.8 Hands-on Labs — Security
- [ ] Implement RBAC + Pod Security Standards + Network Policies for multi-tenant cluster
- [ ] Set up Falco for runtime security monitoring
- [ ] Deploy Vault with Kubernetes auth and secret injection
- [ ] Implement image signing with cosign + verification admission controller
- [ ] Set up security scanning pipeline (Trivy + Checkov + CodeQL)
- [ ] Configure OPA/Gatekeeper policies
- [ ] Implement mTLS with cert-manager
- [ ] Set up AWS Security Hub with automated remediation

---

## Section 12 — Databases

### 12.1 Relational Databases
- **ACID**: Atomicity (all or nothing), Consistency (valid state), Isolation (concurrent transactions), Durability (committed = permanent)
- **PostgreSQL**: WAL (Write-Ahead Log), MVCC (Multi-Version Concurrency Control), vacuum (reclaim dead tuples), connection pooling (PgBouncer)
- **MySQL/Aurora**: InnoDB engine, B-tree indexes, replication (binlog), Aurora 6-way storage replication
- **Indexing**: B-tree (default, range queries), hash (equality), GiST/GIN (full-text, JSON, arrays), partial indexes, composite indexes, covering indexes (index-only scans)
- **Query optimization**: `EXPLAIN ANALYZE`, sequential scan vs index scan, join strategies (nested loop, hash join, merge join), query planner statistics

### 12.2 Non-Relational Databases
- **DynamoDB**: partition key (hash), sort key (range), GSI/LSI, single-table design, on-demand vs provisioned, DAX caching, Streams, global tables
- **Redis**: data structures (strings, hashes, lists, sets, sorted sets), persistence (RDB snapshots, AOF log), clustering (hash slots), sentinel (HA), pub/sub, eviction policies (allkeys-lru, volatile-lru)
- **MongoDB**: document model (BSON), sharding (shard key selection), replica sets (primary + secondaries + arbiter), aggregation pipeline
- **Elasticsearch**: inverted index, mappings, analyzers (standard, keyword), cluster (nodes, shards, replicas), query DSL

### 12.3 Database Operations
- **Backup**: logical (pg_dump/mysqldump), physical (snapshots, EBS snapshot), continuous (WAL archiving, binlog), PITR (Point-In-Time Recovery)
- **Replication**: synchronous (strong consistency, higher latency), asynchronous (eventual consistency, lower latency), streaming, logical
- **High availability**: Multi-AZ (RDS), failover (60-120s), read replicas, connection routing (RDS Proxy, PgBouncer)
- **Scaling**: vertical (bigger instance), horizontal read (read replicas), horizontal write (sharding, DynamoDB partitions)
- **Zero-downtime migrations**: expand-and-contract pattern, backward-compatible schema changes, blue-green database deployment
- **Monitoring**: slow query logs, connection counts, replication lag, disk IOPS, cache hit ratio

### 12.4 Vector Databases (AI-Relevant)
- **Purpose**: similarity search on high-dimensional vectors (embeddings)
- **Options**: Pinecone (managed), Weaviate (open-source), Qdrant (Rust, fast), Milvus (scalable), pgvector (PostgreSQL extension), ChromaDB (simple)
- **Indexing**: HNSW (Hierarchical Navigable Small World — fast, memory-heavy), IVF (Inverted File Index — disk-friendly), PQ (Product Quantization — compression)
- **Distance metrics**: cosine similarity, euclidean (L2), dot product
- **Operations**: dimension selection (tradeoff: accuracy vs speed), batch ingestion, index tuning, metadata filtering

### 12.5 Failure Scenarios — Databases
- **Replication lag**: high write volume, slow replica, network issues → monitor lag metric, route reads appropriately
- **Connection exhaustion**: too many connections → connection pooling (PgBouncer/RDS Proxy), increase max_connections, find connection leaks
- **Slow queries**: `EXPLAIN ANALYZE`, missing indexes, table bloat (PostgreSQL vacuum), statistics stale → add indexes, vacuum, analyze
- **Data corruption**: backup restore, PITR, checksums, WAL replay
- **Disk full**: alert before full, expand EBS, archive old data, partition tables
- **Split brain**: fencing mechanisms, quorum-based decisions, STONITH (Shoot The Other Node In The Head)

### 12.6 Interview Questions — Databases
- Explain ACID properties. How does PostgreSQL implement them?
- How does Aurora achieve high availability?
- When would you use DynamoDB vs PostgreSQL?
- Explain database replication strategies.
- How do you handle zero-downtime database migrations?
- What is connection pooling and why is it important?
- How do you optimize a slow SQL query?
- Explain DynamoDB single-table design.
- When to run a database on Kubernetes vs managed service?
- What are vector databases and when to use one?
- How do you handle database backup and disaster recovery?
- Explain PostgreSQL MVCC.
- How do you scale a relational database horizontally?
- What is replication lag and how to handle it?
- Design a database architecture for a high-traffic application.

### 12.7 Hands-on Labs — Databases
- [ ] Set up PostgreSQL with streaming replication and failover
- [ ] Deploy Aurora with read replicas and test failover
- [ ] Implement DynamoDB single-table design with GSI
- [ ] Set up Redis cluster with sentinel on Kubernetes
- [ ] Implement zero-downtime database migration
- [ ] Set up pgvector for similarity search
- [ ] Configure automated backups with PITR
- [ ] Monitor database performance with slow query analysis

---

## Section 13 — Reliability Engineering

### 13.1 SRE Fundamentals
- **SRE principles**: embracing risk (error budgets), SLOs as contract, eliminating toil, monitoring, automation, release engineering
- **Error budgets**: 100% - SLO = error budget, spend on feature velocity or reliability
- **Toil**: manual, repetitive, automatable, no lasting value — target <50% toil
- **Service tiers**: Tier 1 (revenue-critical, 99.99%), Tier 2 (important, 99.9%), Tier 3 (internal, 99%)

### 13.2 SLIs, SLOs, SLAs
- **SLI**: measurable metric — availability (successful requests / total), latency (p99 < 200ms), throughput, error rate
- **SLO**: target for SLI — "99.9% of requests complete successfully" (30-day rolling window)
- **SLA**: contractual SLO with penalties — "99.9% uptime, else credit"
- **Error budget**: 99.9% SLO = 0.1% error budget = 43.2 min/month downtime allowed
- **Burn rate**: how fast error budget is consumed — 1× = steady, 10× = fast burn (alert!)
- **Multi-window alerts**: fast burn (2% budget in 1h → page) + slow burn (5% budget in 6h → ticket)

### 13.3 High Availability
- **Availability math**: 99.9% = 8.76h/year, 99.99% = 52.6min/year, 99.999% = 5.26min/year
- **Serial dependencies**: A (99.9%) → B (99.9%) = 99.8% combined
- **Parallel redundancy**: A || A = 1 - (0.001 × 0.001) = 99.9999%
- **Multi-AZ**: resources across 2-3 AZs, automatic failover, synchronous replication
- **Multi-region**: async replication, DNS-based failover (Route 53), higher complexity, data consistency challenges
- **DR strategies**: backup/restore (RPO/RTO: hours), pilot light (RPO: minutes, RTO: minutes), warm standby (RPO/RTO: seconds-minutes), active-active (near-zero)

### 13.4 Fault Tolerance Patterns
- **Circuit breaker**: closed (normal) → open (fail fast) → half-open (test); configurable failure threshold and timeout
- **Retry**: exponential backoff with jitter, retry budget (% of total requests), idempotency requirement
- **Bulkhead**: isolate failure domains, separate thread pools/connection pools per dependency
- **Timeout**: connection timeout, read timeout, deadline propagation across service calls
- **Graceful degradation**: feature flags, cached responses, reduced functionality under load
- **Rate limiting**: token bucket, sliding window, distributed rate limiting (Redis)
- **Load shedding**: drop low-priority requests when overloaded, return 503 with Retry-After
- **Chaos engineering**: Chaos Monkey, Litmus, Gremlin — verify fault tolerance in production, game days

### 13.5 Incident Management
- **Lifecycle**: detect → triage → communicate → mitigate → resolve → postmortem
- **Severity levels**: SEV1 (critical outage) → SEV4 (minor issue)
- **Roles**: Incident Commander, Communications Lead, Operations Lead, Subject Matter Experts
- **Communication**: status page (Statuspage.io), stakeholder updates, war room (Slack channel)
- **Postmortem**: blameless, timeline, root cause, contributing factors, action items (prevent, detect, mitigate)
- **On-call**: rotation schedules, escalation policies, runbooks, handoff procedures, compensation

### 13.6 Interview Questions — Reliability
- What is an error budget and how do you use it?
- Design SLOs for a real-time API service.
- How do you implement the circuit breaker pattern?
- Explain high availability vs fault tolerance.
- Design a system for 99.99% availability.
- What is chaos engineering and how do you practice it?
- Walk through your incident response process.
- How do you write a blameless postmortem?
- Explain thundering herd problem and solutions.
- How do you handle cascading failures?
- What is graceful degradation? Give examples.
- How do you plan capacity for a growing service?
- Explain multi-window, multi-burn-rate alerting.
- How do you reduce toil?
- Design DR strategy for a multi-region application.

### 13.7 Hands-on Labs — Reliability
- [ ] Define SLIs/SLOs and implement error budget alerting
- [ ] Implement circuit breaker and retry patterns
- [ ] Run chaos engineering experiments with Litmus
- [ ] Set up multi-AZ deployment with automatic failover
- [ ] Conduct tabletop incident response exercise
- [ ] Implement load testing with k6 and capacity planning
- [ ] Set up multi-window burn rate alerting
- [ ] Write blameless postmortem for simulated outage

---

## Section 14 — AI Infrastructure

### 14.1 AI/ML Fundamentals for Infrastructure Engineers
- **ML workflow**: data preparation → training → evaluation → deployment → monitoring → retraining
- **Training vs Inference**: training = GPU-intensive, batch, hours-days; inference = latency-sensitive, real-time or batch
- **Model types**: classification, regression, NLP (transformers), computer vision (CNNs), generative AI (LLMs, diffusion)
- **Model formats**: PyTorch (.pt), ONNX (interoperable), TensorRT (NVIDIA optimized), SafeTensors (safe serialization), GGUF (quantized for CPU)
- **Model sizes**: 7B params ≈ 14GB FP16, 70B ≈ 140GB FP16, 405B ≈ 810GB FP16 — drives GPU memory requirements

### 14.2 GPU Infrastructure

#### GPU Architecture Basics
- **CUDA cores**: general-purpose parallel processing
- **Tensor cores**: specialized for matrix multiplication (mixed-precision: FP16/BF16/INT8/FP8)
- **GPU memory (HBM)**: High Bandwidth Memory, determines max model size per GPU
- **NVLink**: high-speed GPU-to-GPU interconnect (900 GB/s on H100), critical for multi-GPU inference/training
- **NVSwitch**: connects all GPUs in a node (full mesh), H100 DGX: 8 GPUs fully connected

#### AWS GPU Instances
- **P4d/P4de** (A100): 8× A100 40/80GB, 400 Gbps EFA, training workhorse
- **P5** (H100): 8× H100 80GB, 3200 Gbps EFA, latest generation training
- **G5** (A10G): 1-8× A10G 24GB, cost-effective inference
- **G6** (L4): next-gen inference, better perf/$ than G5
- **Inf2** (Inferentia2): custom inference chip, best cost for transformer inference
- **Trn1** (Trainium): custom training chip, cost-effective alternative to GPU training

#### GPU in Kubernetes
- **NVIDIA device plugin**: exposes GPUs as schedulable resources (`nvidia.com/gpu`)
- **GPU Operator**: automates driver installation, container toolkit, DCGM monitoring, GFD (GPU Feature Discovery)
- **Resource requests**: `resources: { limits: { nvidia.com/gpu: 1 } }` — GPU is a limit-only resource (no fractional sharing by default)
- **Time-slicing**: share GPU across pods (virtual GPUs), reduces cost for dev/test, increases latency
- **MIG (Multi-Instance GPU)**: A100/H100 hardware partitioning into isolated GPU instances (1g.5gb, 2g.10gb, 3g.20gb, 7g.40gb)
- **MPS (Multi-Process Service)**: CUDA-level GPU sharing, higher utilization, no memory isolation

#### GPU Monitoring
- `nvidia-smi`: GPU utilization, memory, temperature, power, processes
- **DCGM exporter**: Prometheus metrics for GPU health, utilization, memory, temperature, ECC errors
- **Key metrics**: GPU utilization %, memory utilization %, temperature, power draw, SM clock, memory clock

### 14.3 Training Infrastructure
- **Distributed training**:
  - Data parallelism: replicate model, split data, average gradients (PyTorch DDP, FSDP)
  - Model parallelism: split model across GPUs (too large for single GPU)
  - Tensor parallelism: split individual layers across GPUs (Megatron-LM)
  - Pipeline parallelism: split layers sequentially across GPUs, micro-batching
- **NCCL**: NVIDIA Collective Communications Library, all-reduce, all-gather, optimized for NVLink/EFA
- **EFA**: Elastic Fabric Adapter, RDMA-like networking, bypasses kernel, required for multi-node NCCL
- **Checkpointing**: save model state periodically to S3, resume after interruption, distributed checkpointing for large models
- **Storage for training**: FSx Lustre (high-throughput parallel FS), S3 data loading (boto3, smart_open), dataset caching on NVMe
- **Training on K8s**: Kubeflow Training Operator (PyTorchJob, MPIJob), volcano scheduler (gang scheduling)
- **Spot training**: use spot instances (60-90% savings), checkpoint frequently, automatic resume on interruption

### 14.4 Inference Infrastructure

#### Model Serving Frameworks
- **Triton Inference Server** (NVIDIA): multi-framework (PyTorch, TensorFlow, ONNX, TensorRT), dynamic batching, model ensemble, concurrent model execution, metrics
- **vLLM**: LLM-specific, PagedAttention (efficient KV cache memory management), continuous batching, tensor parallelism, high throughput
- **TGI** (Text Generation Inference): HuggingFace, streaming responses, quantization support, simple deployment
- **KServe**: Kubernetes-native, InferenceService CRD, autoscaling, canary rollouts, transformers/explainers
- **Ray Serve**: general-purpose, Ray ecosystem integration, dynamic batching, multi-model composition

#### Inference Optimization
- **Batching**: static (fixed batch), dynamic (wait for requests up to timeout), continuous (LLM-specific, process tokens as they arrive)
- **Quantization**: FP32 → FP16/BF16 (2× savings), INT8 (4× savings), INT4 (8× savings), FP8 — tradeoff: size/speed vs accuracy
  - Post-training quantization: GPTQ, AWQ, bitsandbytes
  - Calibration-based: representative data sample for optimal quantization parameters
- **Model compilation**: TensorRT (NVIDIA), torch.compile (PyTorch 2.0), ONNX Runtime — operator fusion, memory optimization
- **KV cache optimization**: PagedAttention (vLLM), prefix caching, KV cache eviction strategies
- **Speculative decoding**: use small draft model to predict tokens, verify with large model in batch — increases throughput

#### Autoscaling Inference
- **Metrics**: GPU utilization, request queue depth, inference latency (p99), tokens per second
- **KEDA**: scale on custom metrics (SQS queue depth, Prometheus GPU utilization), scale to zero
- **Knative**: serverless scaling, scale from/to zero, concurrency-based autoscaling
- **Karpenter**: GPU node provisioning, spot GPU instances, consolidation

### 14.5 Model Deployment Patterns
- **Online inference**: REST/gRPC API, low latency (<100ms), autoscaling, health checks
- **Batch inference**: scheduled Jobs/CronJobs, process large datasets, cost-optimized (spot instances)
- **Streaming inference**: Kafka/Kinesis → model → output stream, real-time predictions on events
- **Multi-model serving**: multiple models on same GPU, model loading/unloading, shared memory, priority queues
- **A/B testing**: traffic splitting (Istio/Argo Rollouts), model version comparison, statistical significance
- **Shadow deployment**: mirror live traffic to new model, compare outputs, no user impact

### 14.6 Failure Scenarios — AI Infrastructure
- **GPU OOM**: model too large for GPU memory → quantize, use tensor parallelism, reduce batch size, offload to CPU
- **Training job failure**: checkpoint recovery, spot interruption → automatic resume, NCCL timeout → check EFA/SG config
- **Inference latency spike**: cold start (model loading), batch queue full, GPU throttling (thermal) → pre-warm, tune batching, monitor temperature
- **NCCL errors**: security group blocking EFA ports, wrong placement group, NCCL version mismatch → check SG rules, use cluster placement group
- **Model loading failure**: corrupted artifact, S3 timeout, incompatible format → verify checksum, retry with backoff, validate format

### 14.7 Interview Questions — AI Infrastructure
- Design inference infrastructure for serving LLMs on Kubernetes.
- Explain GPU scheduling in K8s. What is MIG?
- How does dynamic batching improve throughput?
- Data parallelism vs model parallelism — explain.
- How would you autoscale GPU inference?
- Explain vLLM PagedAttention and why it matters.
- How do you handle model versioning and rollback?
- Cost optimization strategies for GPU workloads?
- How to set up multi-node training on K8s with EFA?
- Explain quantization and its tradeoffs.
- How do you monitor GPU performance?
- Design an ML platform on K8s for 50 data scientists.
- How to handle spot interruptions during training?
- What is KServe and how does it compare to Triton?
- How to implement A/B testing for ML models?

### 14.8 Hands-on Labs — AI Infrastructure
- [ ] Deploy NVIDIA GPU Operator on EKS with GPU nodes
- [ ] Set up Triton Inference Server with dynamic batching
- [ ] Deploy vLLM for LLM serving with autoscaling
- [ ] Implement distributed training with PyTorch DDP on K8s
- [ ] Set up KServe with canary model rollout
- [ ] Configure KEDA autoscaler with GPU metrics
- [ ] Deploy Ray cluster for distributed inference
- [ ] Set up GPU monitoring with DCGM + Datadog
- [ ] Configure spot GPU instances with checkpointing

---

## Section 15 — LLMOps and RAG

### 15.1 LLM Fundamentals for Infrastructure
- **Transformer architecture**: self-attention, KV cache (stores past computations), context window (max input tokens)
- **Model sizes and requirements**: 7B ≈ 1 GPU (A10G), 13B ≈ 1 GPU (A100), 70B ≈ 4 GPUs (A100), 405B ≈ 8+ GPUs (H100)
- **Inference characteristics**: autoregressive (one token at a time), TTFT (Time To First Token), TPS (Tokens Per Second), total latency
- **Tokenization**: BPE (Byte Pair Encoding), SentencePiece — token ≈ 4 characters English, cost per token

### 15.2 LLM Serving Infrastructure
- **vLLM**: PagedAttention (virtual memory for KV cache, no wasted memory), continuous batching, tensor parallelism across GPUs
- **TGI**: HuggingFace serving, streaming SSE, quantization, simple Docker deployment
- **TensorRT-LLM**: NVIDIA optimized, FP8 support, inflight batching, highest throughput on NVIDIA GPUs
- **Key optimizations**:
  - Continuous batching: process new requests while generating tokens for existing ones
  - PagedAttention: allocate KV cache in non-contiguous blocks (like OS virtual memory), near-zero waste
  - Flash Attention: memory-efficient attention computation, O(N) memory instead of O(N²)
  - Speculative decoding: draft model generates candidate tokens, target model verifies in batch
  - Prefix caching: reuse KV cache for shared prompt prefixes (system prompts)
- **Quantization**: GPTQ (post-training, 4-bit), AWQ (activation-aware, 4-bit), FP8 (hardware-native on H100), bitsandbytes (simple)
- **Streaming**: SSE (Server-Sent Events) for token-by-token output, gRPC streaming

### 15.3 RAG (Retrieval-Augmented Generation)

#### Architecture
- **Flow**: user query → embed query → search vector DB → retrieve relevant chunks → augment prompt with context → generate response

#### Components
- **Embedding models**: sentence-transformers, OpenAI text-embedding-3-small/large, Cohere embed — convert text to vectors
- **Vector databases**: Pinecone (managed, simple), Weaviate (open-source, hybrid search), Qdrant (Rust, fast), pgvector (PostgreSQL, simple to start)
- **Chunking**: fixed-size (512 tokens), recursive (split by separators), semantic (topic-based), document-aware (headers, sections)
- **Retrieval**: dense retrieval (vector similarity), sparse retrieval (BM25/keyword), hybrid (combine both), re-ranking (cross-encoder for precision)

#### Advanced RAG
- **Query transformation**: HyDE (Hypothetical Document Embedding), multi-query (rephrase for diversity)
- **Contextual compression**: extract only relevant portions from retrieved chunks
- **Parent-document retrieval**: store small chunks, retrieve parent documents for context
- **Re-ranking**: cross-encoder models score query-document pairs, more accurate than bi-encoder
- **Evaluation**: RAGAS framework — faithfulness, answer relevance, context precision, context recall

### 15.4 LLM Infrastructure Patterns
- **API Gateway for LLMs**: rate limiting per user/key, authentication, model routing, fallback between providers
- **LLM proxy**: LiteLLM (unified API across providers), model abstraction, automatic failover
- **Semantic caching**: cache responses for semantically similar queries (vector similarity on query embeddings), reduce cost and latency
- **Guardrails**: input validation (prompt injection detection), output filtering (PII, toxicity, off-topic), content safety
- **Cost management**: token counting per request, budget enforcement per team/user, route simple tasks to smaller models

### 15.5 LLMOps Pipeline
- **Fine-tuning infrastructure**: LoRA (Low-Rank Adaptation, <1% parameters), QLoRA (quantized LoRA), full fine-tuning on K8s
- **Model evaluation**: benchmarks (MMLU, HumanEval), human evaluation, automated evaluation (LLM-as-judge)
- **Deployment pipeline**: evaluate → stage → canary deploy → monitor → promote to production
- **Monitoring**: token usage (cost), TTFT, TPS, error rates, quality metrics (user feedback, automated checks)
- **Feedback loops**: collect user thumbs up/down, RLHF data pipeline, continuous improvement

### 15.6 Scaling LLM Infrastructure
- **Horizontal**: multiple model replicas behind load balancer, request routing based on load
- **Vertical**: larger GPU instances, multi-GPU tensor parallelism
- **Multi-model**: model multiplexing on shared GPUs, cold start optimization (keep warm pool), priority queues (premium vs free tier)
- **Cost optimization**: spot instances, reserved capacity, smaller models for simple tasks, caching
- **Queue-based**: async inference via SQS/Kafka for non-real-time use cases, batch processing

### 15.7 Failure Scenarios — LLMOps
- **Hallucination**: RAG grounding, output validation, citation verification
- **High latency**: KV cache issues, model too large, insufficient GPUs, batching misconfigured
- **OOM during inference**: context window too large, batch too large → quantize, limit context, adjust batch
- **Vector DB slow**: index not optimized, high dimensionality, complex filters → rebuild index, reduce dimensions, pre-filter
- **Embedding drift**: model update changes vector space → re-index all documents
- **RAG irrelevant results**: chunking too large/small, embedding quality poor, retrieval threshold too low

### 15.8 Interview Questions — LLMOps
- Design LLM serving infrastructure for a production application.
- How does RAG work end to end? Design the infrastructure.
- Explain continuous batching and PagedAttention.
- How to autoscale LLM inference based on demand?
- Tradeoffs between quantization methods?
- How to handle LLM hallucinations in production?
- Design a RAG pipeline with hybrid search and re-ranking.
- How to monitor LLM performance in production?
- Tensor parallelism vs pipeline parallelism?
- How to optimize cost for LLM inference?
- What is semantic caching?
- How to handle model versioning for LLMs?
- Design multi-model serving with fallback.
- How to evaluate RAG quality?
- Infrastructure needed for fine-tuning LLMs?

### 15.9 Hands-on Labs — LLMOps
- [ ] Deploy vLLM with open-source LLM (LLaMA/Mistral) on K8s with GPU
- [ ] Build RAG pipeline with vector DB, embedding model, and LLM
- [ ] Implement semantic caching for LLM responses
- [ ] Set up autoscaling for LLM inference (queue + GPU metrics)
- [ ] Deploy multiple model versions with canary rollout
- [ ] Implement guardrails for LLM output (PII filtering)
- [ ] Set up LLM monitoring dashboard (TTFT, TPS, token usage)
- [ ] Build hybrid search RAG with BM25 + dense + re-ranking
- [ ] Implement LoRA fine-tuning pipeline on K8s
- [ ] Set up LLM proxy with provider failover

---

## Section 16 — Backend Engineering

### 16.1 API Design
- **REST**: resources (nouns, not verbs), HTTP methods (GET/POST/PUT/PATCH/DELETE), status codes, HATEOAS, versioning (URI `/v1/`, header `Accept-Version`, query param)
- **gRPC**: Protocol Buffers (schema, code generation), streaming (unary, server, client, bidirectional), deadline propagation, error codes (OK, CANCELLED, DEADLINE_EXCEEDED, etc.)
- **GraphQL**: schema (types, queries, mutations), resolvers, N+1 problem (DataLoader batching), subscriptions (WebSocket)
- **API Gateway**: rate limiting (token bucket, sliding window), authentication (JWT, API key), routing, throttling, request/response transformation
- **Webhooks**: event-driven, HTTP callbacks, retry with backoff, signature verification (HMAC)

### 16.2 Concurrency & Parallelism
- **Threads vs Processes**: shared memory (threads) vs isolated (processes), GIL in Python
- **Goroutines**: lightweight (2KB stack), M:N scheduling, channels for communication
- **Async/Await**: event loop (Python asyncio, Node.js), non-blocking I/O, cooperative multitasking
- **Synchronization**: mutex, RWLock, semaphore, channels, atomic operations
- **Patterns**: producer-consumer, worker pool, fan-out/fan-in, pipeline
- **Pitfalls**: race conditions, deadlocks (mutual exclusion + hold-and-wait + no preemption + circular wait), livelocks, starvation

### 16.3 Scaling Patterns
- **Horizontal scaling**: stateless services behind load balancer, shared-nothing architecture
- **Connection pooling**: database (PgBouncer, RDS Proxy), HTTP (keep-alive), gRPC (multiplexing)
- **Caching**: local (in-process), distributed (Redis), CDN; invalidation: TTL, event-driven, write-through, cache-aside
- **Async processing**: message queues (SQS, Kafka, RabbitMQ), decouple producers and consumers, handle spikes
- **Backpressure**: rate limit at ingress, bounded queues, load shedding (503 Retry-After), reactive streams

### 16.4 Reliability Patterns
- **Circuit breaker**: prevent cascading failures, fail fast when dependency is down
- **Retry**: exponential backoff (base × 2^attempt), jitter (randomize), retry budget (max % of total requests)
- **Timeout**: connection + read timeout, propagate deadlines across services, shorter than caller's timeout
- **Idempotency**: idempotency keys (client-generated UUID), exactly-once via deduplication
- **Graceful shutdown**: SIGTERM → stop accepting new requests → drain in-flight → exit; preStop hook in K8s
- **Health checks**: liveness (process alive?), readiness (can accept traffic?), startup (initialization complete?)

### 16.5 Message Queues & Event Streaming
- **SQS**: standard (at-least-once, unordered) vs FIFO (exactly-once, ordered), visibility timeout, dead-letter queues, long polling
- **Kafka**: topics, partitions (parallelism unit), consumer groups (load distribution), offsets (position tracking), retention, compaction
- **Event-driven architecture**: event sourcing (store events, derive state), CQRS (separate read/write models), saga pattern (distributed transactions)

### 16.6 Interview Questions — Backend Engineering
- Design a rate limiter. What algorithms?
- How do you handle distributed transactions across microservices?
- Explain circuit breaker pattern.
- gRPC vs REST — when to choose each?
- How to implement idempotency in APIs?
- Explain Kafka consumer groups and partition assignment.
- How to handle graceful shutdown in Kubernetes?
- Design async processing pipeline with SQS.
- What is backpressure and how to implement it?
- How to handle cache invalidation in distributed systems?
- Explain event sourcing and CQRS.
- How to debug a memory leak in production?
- Design an API gateway for microservices.
- Connection pooling for databases — why and how?
- Sync vs async communication between services — tradeoffs?

### 16.7 Hands-on Labs — Backend Engineering
- [ ] Build REST API with rate limiting, circuit breaker, and graceful shutdown
- [ ] Implement Kafka producer/consumer with consumer groups
- [ ] Set up SQS with dead-letter queue and async workers
- [ ] Implement distributed caching with Redis
- [ ] Build gRPC service with streaming and deadlines
- [ ] Implement health check endpoints for Kubernetes

---

## Section 17 — DevEx and Platform Engineering

### 17.1 Platform Engineering Fundamentals
- **Internal Developer Platform (IDP)**: self-service infrastructure for development teams
- **Platform team**: builds tools and abstractions, enables stream-aligned teams, reduces cognitive load
- **Golden paths**: recommended, paved road — opinionated defaults, not mandated
- **Developer experience**: fast CI/CD, self-service, clear documentation, minimal operational burden
- **DORA metrics**: deployment frequency, lead time for changes, MTTR (Mean Time To Recovery), change failure rate

### 17.2 Internal Developer Platform
- **Backstage** (Spotify): software catalog, scaffolder (project templates), TechDocs, plugin ecosystem
- **Service catalog**: discover services, ownership, API docs, dependencies, health status
- **Project templates**: standardized service creation with CI/CD, monitoring, security baked in
- **Platform APIs**: Terraform modules, Helm charts, Crossplane compositions — infrastructure as self-service

### 17.3 CI/CD Optimization
- **Build time**: caching (dependencies, Docker layers, build cache), parallelization, incremental builds
- **Test optimization**: test splitting, parallel execution, flaky test quarantine, test impact analysis
- **Feedback loop**: target <10 min from commit to PR status check
- **DORA metrics tracking**: measure and improve deployment frequency, lead time, MTTR, change failure rate

### 17.4 Developer Self-Service
- **Namespace provisioning**: automated creation with RBAC, quotas, network policies, LimitRanges
- **Ephemeral environments**: per-PR preview environments, ArgoCD ApplicationSets, auto-cleanup
- **Secret management**: self-service via External Secrets Operator, developer doesn't touch K8s secrets directly
- **Database provisioning**: self-service via operators or Terraform modules

### 17.5 Automation & Toil Reduction
- **Runbook automation**: automated remediation for common incidents
- **ChatOps**: Slack bots for deployments, infrastructure actions, incident management
- **Policy automation**: OPA/Gatekeeper, Kyverno — enforce standards without human review
- **GitOps automation**: ApplicationSets, auto-discovery of new services

### 17.6 Interview Questions — DevEx
- What is an Internal Developer Platform and why build one?
- How do you measure developer experience?
- Explain DORA metrics and how to improve them.
- How would you design a self-service infrastructure platform?
- What are golden paths?
- How to reduce CI/CD build times?
- What is Backstage and how does it help?
- Balance standardization vs flexibility?
- Design ephemeral environments for PR testing.
- How to handle platform adoption without mandating it?

### 17.7 Hands-on Labs — DevEx
- [ ] Set up Backstage with service catalog and scaffolder
- [ ] Create reusable CI/CD workflows
- [ ] Implement ephemeral PR environments with ArgoCD
- [ ] Build self-service namespace provisioning
- [ ] Set up DORA metrics tracking

---

## Section 18 — FinOps

### 18.1 FinOps Fundamentals
- **FinOps principles**: inform (visibility), optimize (reduce waste), operate (governance)
- **Cloud cost model**: on-demand, reserved, spot, committed use discounts
- **Cost allocation**: tagging strategy, cost centers, showback (visibility) vs chargeback (billing)
- **FinOps lifecycle**: inform → optimize → operate → repeat

### 18.2 AWS Cost Optimization
- **Compute**: Savings Plans (flexible), RIs (specific), Spot (fault-tolerant), right-sizing (Compute Optimizer)
- **Storage**: S3 Intelligent-Tiering, lifecycle policies, gp3 over gp2, delete unused EBS/snapshots
- **Networking**: VPC endpoints (avoid NAT GW $0.045/GB), minimize cross-AZ data transfer ($0.01/GB), CloudFront for S3
- **Database**: Aurora Serverless v2 for variable workloads, stop dev instances off-hours, reserved capacity
- **Kubernetes**: Karpenter consolidation, spot node pools, right-size requests/limits, namespace quotas

### 18.3 Kubernetes Cost Optimization
- **Right-sizing**: VPA recommendations for requests, avoid over-provisioning limits
- **Node efficiency**: Karpenter bin-packing, consolidation (terminate under-utilized nodes), right-size instance types
- **Spot instances**: Karpenter spot, diversify instance types (10+ types), handle interruptions gracefully
- **Cost visibility**: Kubecost/OpenCost per namespace/team/label, showback reports
- **Idle resources**: detect unused PVCs, dangling load balancers, orphaned resources → automated cleanup

### 18.4 Cost Monitoring & Governance
- **AWS Cost Explorer**: daily/monthly trends, service breakdown, filter by tags
- **AWS Budgets**: spending threshold alerts, forecasted budget alerts, automated actions
- **Cost Anomaly Detection**: ML-based, automatic root cause identification
- **Tagging enforcement**: AWS Config rules (`required-tags`), SCPs, CI/CD checks (Checkov)
- **FinOps dashboards**: per-team, per-service, per-environment cost visibility

### 18.5 Interview Questions — FinOps
- How to implement cost allocation in multi-team K8s cluster?
- What are the biggest hidden costs in AWS?
- Reserved Instances vs Savings Plans vs Spot — tradeoffs?
- How does Karpenter reduce costs?
- Cost optimization without affecting reliability?
- Design a tagging strategy for multi-account AWS.
- How to reduce data transfer costs?
- Showback vs chargeback?
- How to identify and eliminate idle resources?
- Set up cost governance for 100-engineer org?

### 18.6 Hands-on Labs — FinOps
- [ ] Set up AWS Cost Explorer with custom reports
- [ ] Implement Kubecost for namespace cost allocation
- [ ] Configure Karpenter consolidation + spot for savings
- [ ] Set up AWS Budgets with alerts
- [ ] Implement tagging enforcement with AWS Config
- [ ] Create FinOps dashboard per team/environment

---

## Section 19 — Architecture Design

### 19.1 Distributed Systems Fundamentals
- **CAP theorem**: Consistency, Availability, Partition tolerance — during partition, choose CP (consistent but unavailable) or AP (available but inconsistent)
- **PACELC**: if Partition → A or C; Else → Latency or Consistency
- **Consistency models**: strong (linearizable), eventual (converges), causal (respects causality), read-your-writes, monotonic reads
- **Consensus**: Raft (etcd, Consul), Paxos, ZAB (ZooKeeper) — leader election + log replication
- **Distributed clocks**: Lamport timestamps (ordering), vector clocks (causality), hybrid logical clocks
- **Split brain**: quorum (N/2+1), fencing tokens, STONITH

### 19.2 Scalability Patterns
- **Horizontal scaling**: stateless services, data partitioning, load balancing
- **Sharding**: hash-based (consistent hashing), range-based (alphabetical/time), geographic — shard key selection is critical
- **CQRS**: Command Query Responsibility Segregation — separate read (optimized queries) and write (event processing) models
- **Event sourcing**: store events as source of truth, derive current state by replay, append-only log
- **Saga pattern**: distributed transaction as sequence of local transactions + compensating actions
  - Choreography: services react to events (decoupled, complex failure handling)
  - Orchestration: central coordinator manages steps (simpler logic, single point of failure)

### 19.3 Microservices Architecture
- **Microservices vs Monolith**: independent deployment/scaling vs simpler development/debugging; start monolith, extract services when needed
- **Service boundaries**: bounded contexts (DDD), data ownership, team alignment
- **Communication**: sync (REST, gRPC) vs async (events, queues) — prefer async for decoupling
- **Service discovery**: DNS (Kubernetes services), registry (Consul), service mesh
- **Data management**: database per service, no shared databases, data synchronization via events
- **Decomposition**: by business capability, by subdomain, strangler fig pattern (gradual migration)

### 19.4 High Availability Architecture
- **Active-active**: both sites serve traffic, data conflict resolution needed (last-writer-wins, CRDTs)
- **Active-passive**: standby site, failover on failure, simpler but slower recovery
- **DR metrics**: RPO (max data loss), RTO (max downtime)
- **DR strategies**:
  - Backup/restore: RPO hours, RTO hours, cheapest
  - Pilot light: core running, RPO minutes, RTO 10-30 min
  - Warm standby: scaled-down replica, RPO/RTO seconds-minutes
  - Active-active: near-zero RPO/RTO, most complex and expensive

### 19.5 Cloud-Native Patterns
- **12-factor app**: codebase (Git), dependencies (declared), config (env vars), backing services (attached resources), build-release-run, stateless processes, port binding, concurrency (scale out), disposability (fast start/stop), dev-prod parity, logs (event streams), admin processes
- **Sidecar**: auxiliary container for cross-cutting concerns (logging, proxy, monitoring)
- **Strangler fig**: gradually replace monolith components with microservices
- **Bulkhead**: isolate components to prevent cascading failures
- **Anti-corruption layer**: adapter between new service and legacy system

### 19.6 System Design Interview Patterns
- **URL shortener**: hash function, base62 encoding, cache hot URLs, rate limiting, analytics
- **Distributed cache**: consistent hashing, replication, eviction (LRU), partitioning
- **Notification system**: fan-out, pub-sub, delivery guarantees, rate limiting, multi-channel
- **ML inference platform**: model registry, serving infrastructure, autoscaling, A/B testing, monitoring
- **Container orchestration platform**: scheduling, networking, storage, self-healing, multi-tenancy

### 19.7 Interview Questions — Architecture
- Explain CAP theorem with real-world examples.
- Design a system handling 100K req/s.
- How to migrate monolith to microservices?
- Design multi-region active-active architecture.
- Explain saga pattern — orchestration vs choreography.
- How to handle distributed transactions?
- Design real-time data pipeline for ML feature serving.
- Explain event sourcing tradeoffs.
- Design infrastructure for a new AI product.
- Design K8s platform for 200 microservices.
- How to determine service boundaries?
- Explain consistency models and when to use each.
- Design DR for a critical financial application.
- How to handle hot partitions in sharded DB?
- Design autoscaling for GPU inference.

### 19.8 Hands-on Labs — Architecture
- [ ] Design and document production architecture for 3-tier app on AWS
- [ ] Implement saga pattern with SQS/SNS
- [ ] Design multi-AZ K8s architecture with failover testing
- [ ] Implement distributed cache with consistent hashing
- [ ] Present system design for ML inference platform (whiteboard)

---

## Section 20 — Debugging and Incident Response

### 20.1 Production Debugging Methodology
- **Scientific method**: observe → hypothesize → test → conclude
- **Top-down**: user impact → service → pod → container → host → network
- **Bottom-up**: infra metrics → service metrics → app logs → traces
- **Comparison-based**: what changed? deployment, config, traffic, dependency
- **USE** for resources: Utilization, Saturation, Errors
- **RED** for services: Rate, Errors, Duration

### 20.2 Kubernetes Debugging
- **Pod debugging**:
  - `kubectl describe pod` — events, conditions, scheduling
  - `kubectl logs <pod> [-c container] [--previous]` — current/previous logs
  - `kubectl exec -it <pod> -- /bin/sh` — interactive shell
  - `kubectl debug <pod> --image=busybox` — ephemeral debug container
  - `kubectl get events --sort-by=.lastTimestamp` — cluster events
- **Common issues**:
  - Pending: no resources, taints, affinity mismatch, PVC not bound
  - CrashLoopBackOff: app crash, missing config, OOM, probe failure
  - ImagePullBackOff: wrong image, auth, rate limit
  - OOMKilled: memory limit exceeded, leak
  - Terminating: finalizers, PDB, preStop hook
- **Service debugging**: check endpoints, DNS, network policies, kube-proxy
- **Node debugging**: `kubectl describe node`, kubelet logs, disk/memory pressure

### 20.3 Incident Response Framework
- **Detect**: monitoring alerts, customer reports, anomaly detection
- **Triage**: severity (SEV1-4), impact scope, urgency
- **Communicate**: status page, stakeholder updates, war room
- **Mitigate**: rollback, feature flag, traffic shift, scale up
- **Resolve**: root cause fix, verification, monitoring
- **Postmortem**: blameless, timeline, root cause, action items

### 20.4 Incident Command System
- **Incident Commander**: owns incident, coordinates, makes decisions
- **Communications Lead**: status page, stakeholder updates
- **Operations Lead**: executes technical remediation
- **Severity**:
  - SEV1: complete outage, revenue impact, all hands
  - SEV2: significant degradation, on-call + escalation
  - SEV3: partial degradation, limited impact
  - SEV4: minor issue, next business day

### 20.5 Postmortem Best Practices
- **Blameless**: focus on systems, not individuals
- **Timeline**: precise timestamps, actions taken
- **Root cause**: 5 Whys, fishbone diagram
- **Action items**: specific, assigned, time-bound — categorize as prevent/detect/mitigate
- **Follow-through**: track completion, review regularly
- **Share**: publish internally, promote learning

### 20.6 Common Production Incidents
- **Service outage**: check recent deployments, dependencies, infrastructure
- **High latency**: CPU/memory, DB slow queries, network, GC pauses, dependency slowness
- **Memory leak**: growing RSS, OOM kills, heap dumps, profiling
- **Certificate expiry**: openssl check, renew, restart, set up monitoring + auto-renewal
- **Disk full**: find large files, clean logs, expand volume, set alerts
- **DB connection exhaustion**: connection pool config, leak detection, increase pool
- **DNS failure**: CoreDNS pods, upstream DNS, resolv.conf

### 20.7 Debugging Tools Cheat Sheet
- **Kubernetes**: kubectl, k9s, stern (multi-pod logs), kubectx/kubens
- **Networking**: tcpdump, wireshark, mtr, dig, curl, openssl, ss
- **Linux**: strace, perf, flamegraphs, dmesg, journalctl
- **Containers**: docker logs, docker inspect, crictl, nsenter
- **AWS**: CloudWatch Logs Insights, X-Ray, VPC Flow Logs, CloudTrail
- **Observability**: Datadog APM, Grafana, Prometheus

### 20.8 Interview Questions — Debugging
- Walk through debugging a CrashLoopBackOff pod.
- Production service returning 500s — your approach?
- How do you handle a production incident?
- K8s node shows NotReady — causes and debugging?
- EKS pods can't reach external API — debug network path?
- How to write a blameless postmortem?
- Database running slow — investigation process?
- Set up on-call for team of 6?
- Deployment caused 50% latency increase — investigate?
- How to debug OOM kills in K8s?
- Correlate metrics, logs, and traces for intermittent timeouts?
- How to handle alert fatigue?
- Debug TLS certificate error in production?
- Pod running but health checks failing — what to check?
- Roll back vs roll forward during an incident — how to decide?

### 20.9 Hands-on Labs — Debugging
- [ ] Debug 10+ broken K8s deployments (various failure modes)
- [ ] Conduct tabletop incident response exercise
- [ ] Write postmortem for simulated outage
- [ ] Set up on-call rotation with PagerDuty
- [ ] Debug network connectivity with tcpdump + VPC Flow Logs
- [ ] Investigate simulated memory leak with profiling

---

## Section 21 — Interview Preparation

### 21.1 Interview Format Overview
- **Phone screen** (30-45 min): behavioral + technical basics
- **Technical deep dive** (60 min): deep knowledge in 1-2 domains
- **System design** (60 min): architecture design, scalability, tradeoffs
- **Hands-on/live coding** (45-60 min): Terraform, Kubernetes YAML, debugging
- **Behavioral** (30-45 min): STAR format, leadership, collaboration
- **Hiring manager** (30-45 min): culture fit, career goals, team dynamics

### 21.2 System Design Framework
1. **Clarify requirements** (3-5 min): functional, non-functional (scale, latency, availability), constraints
2. **High-level design** (10-15 min): components, data flow, API design
3. **Deep dive** (20-25 min): critical components, data model, scaling strategy
4. **Infrastructure** (10-15 min): cloud services, K8s, networking, CI/CD, monitoring
5. **Tradeoffs & discussion** (5-10 min): alternatives, monitoring, cost, security, failure modes

### 21.3 Behavioral Interview (STAR)
- **Situation**: context, team, project
- **Task**: your specific responsibility
- **Action**: what you did, decisions, challenges overcome
- **Result**: measurable outcome, impact, lessons learned

### 21.4 Common Behavioral Questions
- Tell me about a difficult technical decision you made.
- Describe a production incident you handled.
- How do you handle technical disagreements?
- Time you had to learn new technology quickly.
- How you improved a process or system significantly.
- How you prioritize multiple urgent tasks.
- Time you mentored someone.
- Describe a failure and what you learned.

### 21.5 System Design Questions for Infrastructure
- Design infrastructure for SaaS application serving 1M users.
- Design K8s platform for 100 microservices with CI/CD.
- Design ML inference platform with autoscaling and A/B testing.
- Design multi-region DR strategy for financial application.
- Design developer platform with self-service provisioning.
- Design monitoring and alerting for microservices architecture.
- Design secure, cost-optimized EKS cluster.
- Design RAG pipeline infrastructure for AI product.

### 21.6 Technical Deep Dive Topics
- Kubernetes architecture and operations
- AWS networking and security
- Terraform state management and modules
- CI/CD pipeline design and optimization
- Container security best practices
- Monitoring and observability design
- AI/ML infrastructure and GPU workloads
- Incident response and debugging

### 21.7 Whiteboard Exercises
- Draw K8s networking model (pod-to-pod, pod-to-service, external-to-pod)
- Design VPC architecture for production EKS cluster
- Draw CI/CD pipeline from commit to production
- Design ArgoCD GitOps workflow with environments
- Draw TLS 1.3 handshake flow
- Design monitoring stack (metrics, logs, traces)
- Draw packet flow from client to K8s pod

### 21.8 Hands-on Exercise Topics
- Write Terraform to deploy EKS cluster with VPC
- Write GitHub Actions workflow for container build + deploy
- Debug broken Kubernetes deployment (given cluster with issues)
- Write K8s manifests: deployment, service, ingress, netpol, HPA
- Write Helm chart for microservice
- Configure ArgoCD Application with sync policies
- Write PromQL queries for SLI calculations
- Debug networking issues in EKS cluster

### 21.9 Interview Preparation Checklist
- [ ] Prepare 5-6 STAR stories (leadership, failure, conflict, impact, learning)
- [ ] Practice system design with 45-min timer and diagrams
- [ ] Review all Kubernetes components — explain any in depth
- [ ] Practice live Terraform and K8s YAML writing
- [ ] Review AWS services (EKS, VPC, IAM, S3, RDS)
- [ ] Prepare questions for interviewer about team/stack/culture
- [ ] Do mock interviews with peers
- [ ] Review this roadmap section by section
- [ ] Practice debugging scenarios hands-on
- [ ] Review cost optimization strategies

---

## Section 22 — Goodnotes Stack Mastery

### 22.1 Stack Overview
| Component | Technology |
|-----------|------------|
| Cloud | AWS |
| Orchestration | EKS |
| IaC | Terraform |
| CI/CD | GitHub Actions |
| GitOps | ArgoCD |
| Packaging | Helm |
| Monitoring | Datadog |
| AI | Inference on K8s (GPUs) |

### 22.2 EKS Production Operations
- **Cluster**: managed control plane, managed node groups (on-demand + spot), Karpenter
- **Networking**: VPC CNI + prefix delegation, Calico/Cilium for network policies
- **Upgrades**: control plane (in-place) → add-ons → node groups (rolling/blue-green) → validate
- **Add-ons**: CoreDNS, kube-proxy, VPC CNI, EBS CSI — version compatibility matrix
- **IAM**: IRSA (OIDC → STS → temporary credentials), Pod Identity (newer, simpler)
- **Cost**: Karpenter consolidation, spot instances, right-sizing, Compute Optimizer
- **DR**: Velero backup, multi-AZ node distribution, PDB for availability

### 22.3 Terraform at Scale
- **Modules**: VPC, EKS, IAM, S3 — standardized, versioned, tested
- **State**: S3 + DynamoDB, separate state per environment/component
- **CI/CD**: plan on PR → review → merge → apply (GitHub Actions)
- **Environments**: dev/staging/prod via directories or Terragrunt
- **Policy**: Checkov/tfsec in CI, prevent insecure configurations
- **Operations**: import existing resources, state migration for refactoring

### 22.4 GitHub Actions Production Patterns
- **Pipeline**: build → test → scan → Docker build → ECR push → update GitOps repo
- **OIDC**: GitHub → AWS STS, no long-lived credentials
- **Self-hosted runners**: ARC on EKS, autoscaling, ephemeral
- **Reusable workflows**: shared across repos, standardized CI
- **Caching**: dependency cache, Docker layer cache, build cache
- **Security**: least-privilege permissions, Dependabot, CodeQL

### 22.5 ArgoCD Production Patterns
- **HA deployment**: 3+ replicas, Redis sentinel
- **App of Apps**: bootstrap pattern, manage all apps from one parent
- **ApplicationSets**: multi-cluster, multi-environment deployment
- **Environment promotion**: dev → staging → prod via Git branches or kustomize overlays
- **Secrets**: External Secrets Operator → AWS Secrets Manager
- **Progressive delivery**: Argo Rollouts with Datadog analysis for canary
- **Notifications**: Slack for sync/failure alerts

### 22.6 Helm Production Patterns
- **Chart repository**: OCI registry (ECR)
- **Base chart**: standardized template for all services
- **Environment values**: values-dev.yaml, values-staging.yaml, values-prod.yaml
- **Testing**: helm lint, helm template, ct (chart-testing) in CI
- **Dependencies**: Chart.yaml dependencies, umbrella charts

### 22.7 Datadog Production Setup
- **Agent**: Datadog Operator (DatadogAgent CRD), DaemonSet + Cluster Agent
- **APM**: auto-instrumentation, service map, trace analytics
- **Logs**: stdout collection, pipelines, indexing, archives to S3
- **Metrics**: DogStatsD custom metrics, unified tagging (env, service, version)
- **Dashboards**: service overview, infrastructure, SLO tracking
- **Alerting**: metric/log/APM monitors, composite monitors, SLO alerts
- **K8s**: Container Insights, autodiscovery, live containers

### 22.8 AI Inference on EKS
- **GPU nodes**: P4/G5/G6 instances, NVIDIA GPU Operator
- **Model serving**: vLLM / Triton on EKS with GPU scheduling
- **Autoscaling**: KEDA with GPU utilization / queue depth metrics
- **Deployment**: Helm charts for model servers, ArgoCD for GitOps
- **Monitoring**: DCGM exporter → Datadog, GPU dashboards
- **Cost**: spot GPUs, time-slicing for dev/test, Karpenter GPU provisioning

### 22.9 End-to-End Workflow
1. Developer pushes code to GitHub
2. GitHub Actions: build → test → security scan → Docker build → ECR push
3. GitHub Actions updates Helm values in GitOps repo
4. ArgoCD detects change, syncs to EKS
5. Argo Rollouts: canary deployment with Datadog analysis
6. Datadog monitors: metrics, logs, traces, SLOs
7. On failure: automatic rollback, Slack notification
8. Infra changes: Terraform PR → plan → review → merge → apply

### 22.10 Interview Questions — Goodnotes Stack
- Walk through CI/CD from code commit to production.
- How to manage EKS upgrades with zero downtime?
- Terraform workflow — state, modules, environments?
- ArgoCD — App of Apps, ApplicationSets?
- Secrets in GitOps workflow?
- IRSA — how does it work internally?
- Cost optimization in EKS cluster?
- Monitor K8s workloads with Datadog?
- Deploy ML models to EKS?
- Canary deployment with Argo Rollouts?
- Self-hosted GitHub Actions runners on K8s?
- DR strategy for EKS cluster?
- Security policies in K8s cluster?
- Helm chart versioning and env-specific values?
- Debug production issue using Datadog?

### 22.11 Hands-on Labs — Goodnotes Stack
- [ ] Deploy complete stack: VPC + EKS + ArgoCD + Datadog via Terraform
- [ ] Set up GitHub Actions with OIDC + Docker build + ECR push
- [ ] Configure ArgoCD App of Apps with multi-env promotion
- [ ] Deploy Argo Rollouts canary with Datadog analysis
- [ ] Set up Datadog monitoring (APM, logs, dashboards, SLO alerts)
- [ ] Configure Karpenter with spot + consolidation
- [ ] Deploy vLLM on EKS with GPU nodes + autoscaling
- [ ] Implement External Secrets with AWS Secrets Manager
- [ ] Set up ARC for self-hosted runners on EKS
- [ ] End-to-end: code push → GHA → ECR → ArgoCD → canary → Datadog

---

## Section 23 — Expert Level Mastery

### 23.1 Expert Kubernetes
- **Custom schedulers**: GPU-aware, topology-aware scheduling
- **Operator development**: kubebuilder, controller-runtime, reconciliation loops, status subresource
- **API extensibility**: aggregated API servers, CRDs, conversion webhooks, admission webhooks
- **Multi-cluster**: Cluster API, fleet management, federated services
- **eBPF**: Cilium deep dive, kernel-level networking/security, Hubble observability
- **Performance tuning**: API server tuning (--max-requests-inflight), etcd performance, 5000+ node clusters
- **Internals**: watch mechanism, informers, work queues, controller patterns, client-go

### 23.2 Expert AWS
- **Landing zone**: Control Tower, account factory, SCPs, baseline guardrails
- **Network architecture**: Transit Gateway + inspection VPC, centralized egress, DNS hub
- **Cost architecture**: org-level RI/SP sharing, commitment management
- **Security architecture**: centralized GuardDuty, Security Hub aggregation, cross-account roles
- **Multi-region**: active-active data replication, DynamoDB global tables, Aurora global databases

### 23.3 Expert Infrastructure as Code
- **Platform engineering with Terraform**: self-service modules, internal provider development
- **Crossplane**: K8s-native infrastructure, Compositions, XRDs, claim-based provisioning
- **Policy as Code**: OPA Rego, Sentinel, automated compliance verification
- **GitOps for infrastructure**: Terraform + ArgoCD/Flux, drift reconciliation
- **Testing**: terratest, terraform test, integration testing, chaos testing infra

### 23.4 Expert Observability
- **Observability-driven development**: instrument before deploy, trace-based testing
- **AIOps**: ML anomaly detection, automated root cause analysis, event correlation
- **Custom exporters**: write Prometheus exporters for internal systems
- **eBPF observability**: Pixie, Hubble — kernel-level tracing without app instrumentation
- **Cost-effective**: metrics aggregation, log sampling, trace sampling, retention tiers
- **SLO engineering**: error budget policies, SLO-driven development

### 23.5 Expert AI Infrastructure
- **Large-scale training**: 100+ GPUs, NCCL optimization, topology-aware scheduling, EFA tuning
- **Inference optimization**: TensorRT compilation, Flash Attention, speculative decoding, continuous batching tuning
- **Multi-model serving**: model multiplexing, shared GPU memory, priority scheduling
- **ML platform design**: end-to-end (training → eval → deploy → monitor), self-service for ML engineers
- **Edge inference**: model optimization (quantization, pruning, distillation), ONNX Runtime

### 23.6 Expert Reliability
- **Chaos engineering at scale**: automated experiments, continuous verification in production
- **Global load balancing**: anycast, multi-CDN, traffic engineering
- **Zero-downtime migrations**: database, service, DNS, infrastructure migrations
- **Capacity planning**: queueing theory, Little's Law (L = λW), forecasting models
- **Resilience testing**: game days, DR drills, failover testing — regular cadence

### 23.7 Expert Security
- **Supply chain security**: SLSA Level 3+, hermetic builds, provenance verification
- **Zero trust architecture**: BeyondCorp, identity-based access, micro-segmentation everywhere
- **CSPM**: Cloud Security Posture Management, automated remediation
- **Threat modeling**: STRIDE for infrastructure, attack trees, risk assessment
- **Incident forensics**: container forensics, log analysis, network forensics

### 23.8 Architecture Thinking
- **Technical strategy**: roadmap planning, buy vs build decisions, technology evaluation
- **Platform thinking**: build for internal customers, API-first, composable architecture
- **Team topology**: platform team, stream-aligned teams, enabling teams (Team Topologies book)
- **Technical leadership**: ADRs (Architecture Decision Records), RFC process, design reviews
- **Cost-performance**: right tool for the job, avoid over-engineering, pragmatic solutions

### 23.9 Thought Leadership
- **Writing**: blog posts, internal docs, ADRs, RFCs
- **Speaking**: internal tech talks, conference presentations
- **Open source**: contribute to CNCF projects, build internal tools
- **Mentoring**: junior engineers, cross-team knowledge sharing
- **Community**: Kubernetes SIGs, AWS community builders, meetups

### 23.10 Expert Interview Questions
- Design K8s platform for 500 microservices across 3 regions.
- Implement zero-trust security for K8s platform.
- Design ML training + inference platform for 100 ML engineers.
- Handle major cloud provider outage.
- Design self-service developer platform with golden paths.
- Technical strategy and roadmap planning approach.
- Evaluate and adopt new technologies.
- Design infra scaling from 1K to 1M users.
- Migrate from on-premises to cloud with zero downtime.
- Design observability platform for 10B events/day.
- Implement compliance as code (SOC 2, HIPAA).
- Design multi-tenant K8s with strong isolation.
- Build chaos engineering practice from scratch.
- Incident response for 50-person engineering org.
- Network architecture for multi-region, multi-account AWS.
- Balance developer velocity vs platform stability.
- Design RAG infrastructure serving 10K queries/second.
- Implement SLO-driven development.
- Design infrastructure for real-time AI application.
- Mentor and grow infrastructure engineers.

### 23.11 Expert Hands-on Labs
- [ ] Build custom K8s operator for ML model deployments
- [ ] Design and implement AWS landing zone with Terraform
- [ ] Build self-service platform with Backstage + ArgoCD + Terraform
- [ ] Implement chaos engineering with Litmus and automated game days
- [ ] Build end-to-end ML platform (train → registry → deploy → monitor)
- [ ] Design and deploy multi-region active-active architecture
- [ ] Build custom Prometheus exporter
- [ ] Implement SLSA Level 3 supply chain security
- [ ] Design and deploy global load balancing with DNS failover
- [ ] Conduct full production DR drill

---

> **End of Roadmap**
>
> This roadmap covers Foundations (Level 0) through Expert/Architect level (Level 7) for AI Cloud Infrastructure Engineering roles. Use it as a study guide, interview preparation resource, and career development roadmap.
>
> Total coverage: 23 sections, 200+ interview questions, 100+ hands-on labs, comprehensive failure scenarios and debugging guides.
