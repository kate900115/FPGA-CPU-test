#include <linux/module.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__attribute__((section(".gnu.linkonce.this_module"))) = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif

static const struct modversion_info ____versions[]
__used
__attribute__((section("__versions"))) = {
	{ 0x402de2cc, __VMLINUX_SYMBOL_STR(module_layout) },
	{ 0x23dc6baf, __VMLINUX_SYMBOL_STR(kmalloc_caches) },
	{ 0xd2b09ce5, __VMLINUX_SYMBOL_STR(__kmalloc) },
	{ 0x42dbad94, __VMLINUX_SYMBOL_STR(single_open) },
	{ 0x5868c8aa, __VMLINUX_SYMBOL_STR(nvidia_p2p_free_page_table) },
	{ 0x13e0209a, __VMLINUX_SYMBOL_STR(single_release) },
	{ 0xee232d29, __VMLINUX_SYMBOL_STR(boot_cpu_data) },
	{ 0x661f6662, __VMLINUX_SYMBOL_STR(seq_printf) },
	{ 0x1fb2ecf0, __VMLINUX_SYMBOL_STR(remove_proc_entry) },
	{ 0xc0e73428, __VMLINUX_SYMBOL_STR(seq_read) },
	{ 0x7e399228, __VMLINUX_SYMBOL_STR(nvidia_p2p_put_pages) },
	{ 0xb44ad4b3, __VMLINUX_SYMBOL_STR(_copy_to_user) },
	{ 0x304bc299, __VMLINUX_SYMBOL_STR(PDE_DATA) },
	{ 0xd8e48d2c, __VMLINUX_SYMBOL_STR(misc_register) },
	{ 0x27e1a049, __VMLINUX_SYMBOL_STR(printk) },
	{ 0x1c254be6, __VMLINUX_SYMBOL_STR(nvidia_p2p_get_pages) },
	{ 0x5944d015, __VMLINUX_SYMBOL_STR(__cachemode2pte_tbl) },
	{ 0xbdfb6dbb, __VMLINUX_SYMBOL_STR(__fentry__) },
	{ 0x33753143, __VMLINUX_SYMBOL_STR(kmem_cache_alloc_trace) },
	{ 0x9d534484, __VMLINUX_SYMBOL_STR(proc_create_data) },
	{ 0x50a0d102, __VMLINUX_SYMBOL_STR(seq_lseek) },
	{ 0x37a0cba, __VMLINUX_SYMBOL_STR(kfree) },
	{ 0x65d044b1, __VMLINUX_SYMBOL_STR(remap_pfn_range) },
	{ 0x362ef408, __VMLINUX_SYMBOL_STR(_copy_from_user) },
	{ 0x4ebeff4a, __VMLINUX_SYMBOL_STR(misc_deregister) },
};

static const char __module_depends[]
__used
__attribute__((section(".modinfo"))) =
"depends=nvidia";


MODULE_INFO(srcversion, "835E4F5388FCEDB1857E789");
