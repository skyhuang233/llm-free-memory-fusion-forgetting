use File::Copy qw(copy);

$out_dir = 'out';
$aux_dir = 'out';

END {
  my $src_pdf = "$out_dir/main.pdf";
  my $dst_pdf = "main.pdf";
  if (-e $src_pdf) {
    copy($src_pdf, $dst_pdf);
  }
}
