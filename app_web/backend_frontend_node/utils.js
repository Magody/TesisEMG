function log(context, msg) {

	const d = new Date();
	const ye = new Intl.DateTimeFormat('en', { year: 'numeric' }).format(d);
	const mo = new Intl.DateTimeFormat('en', { month: 'short' }).format(d);
	const da = new Intl.DateTimeFormat('en', { day: '2-digit' }).format(d);

	const date = `${da}-${mo}-${ye}`
	console.log("[" + context + "]|[" + date + "]: ", msg)
}


module.exports = {
  log
}