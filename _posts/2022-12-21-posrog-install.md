---
layout: post
title: Install POSROG V3U8 (PhoenixOS) alongside Solus
subtitle: Systemd Solus Configuration
length: 5
image: install-posrog-banner.jpg
image-caption: Republic of Gamers - The Choice of Champions
---

POSROG, or PhoenixOS for the Republic of Gamers, is an Android operating system with x86 architecture that has been modified from AOSP. It is a light Emulator that focuses on device support, performance, and boosting for gaming applications.

On my low-end AMD e2-9000, I chose to dualboot posrog and solus. This was a relatively simple procedure, but it was distinct because Solus employs the systemd bootloader, whereas posrog installations typically employ grub.

Here, I reflect on the steps I took to complete this task.

### Creating The POSROG Partition

- Boot into Solus and create an EXT4 partition using Gparted.

```
  Label: ANDROID
  Size: 40GB.
```

Nb: Copy the UUID of the partition and save. (To be used later on).

- Mount ANDROID and create directory structure:
  `posrog/data/`

- Download Official [POSROG V3U8 ](https://posrog.com.id/download). (Kernel and architecture chosen to match my computer). Extract.

- Locate the ISO and extract it.

- Copy the following files into posrog directory of ANDROID partition.

  - `posrog`,
  - `ramdisk.img`,
  - `system.img`,

NB: leave data directory empty. It will be used to create the android data files by POSROG during installation.

### Adding Systemd Bootloader Entry for POSROG

And now comes the fun part...

To configure systemd to add an entry for POSROG; Since Solus unmounts the boot partition after booting, it will have to be mounted to edit the systemd configuration files.

Launch a new terminal window and perform the following sequence of operations:

- Mount the EFI System partition which contains the bootloader to `/boot`.

  ```bash
  `sudo mount /dev/sdX# /boot`

  # Where sdX# is the EFI System Partition (ESP).
  ```

- Change directory to `/boot/EFI`. Create subdirectory `posrog`. Copy `initrd.img` and the kernel files from the iso to this folder.

3. Change directory to `/boot/entries`. Add a new file `posrog.current.conf` to create configuration file for posrog. Open the file in nano as su.

4. Add the following lines:

```
title POSROG V3U8 - Android x86
linux /EFI/posrog/kernelxx
initrd/EFI/posrog/initrd.img
options root=PARTUUID=THE_UUID_WE_COPIED_BEFORE rw quiet androidboot.selinux=permissive buildvariant=userdebug acpi_sleep=s3_bios,s3_mode SRC=posrog/
search set=root file /posrog/system.img
```

- Save and exit nano. Navigate back to root directory.
- Unmount the ESP using `sudo umount -R /boot`

### Displaying the Solus boot menu by default on boot

The following command will set the timeout of the boot loader so that it appears by default.

`sudo clr-boot-manager set-timeout 5 && sudo clr-boot-manager update`

### INSTALL POSROG

This is the final step.

Restart your pc and select the POSROG entry from **systemd menu**. Wait for it to boot and follow the GUI prompt to configure POSROG.
When android boots, select Install without WiFi.
