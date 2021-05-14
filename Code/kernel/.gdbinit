define init
	target remote tcp::1234
	b main
	b debug_sign
	c
end

define cur
	print task_list_cur->cpu
	print task_list_cur->status
end

define checkpc
	watch $rip <= 0x300000
end


