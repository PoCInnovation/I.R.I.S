[4mLS[24m(1)                                               User Commands                                               [4mLS[24m(1)

[1mNAME[0m
     ls - list directory contents

[1mSYNOPSIS[0m
     [1mls [22m[[4mOPTION[24m]... [[4mFILE[24m]...

[1mDESCRIPTION[0m
     List  information  about  the  FILEs (the current directory by default).  Sort entries alphabetically if none of
     [1m-cftuvSUX [22mnor [1m--sort [22mis specified.

     Mandatory arguments to long options are mandatory for short options too.

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-a\[1m-a, --all[22m]8;;\
            do not ignore entries starting with .

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-A\[1m-A, --almost-all[22m]8;;\
            do not list implied . and ..

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls--author\[1m--author[22m]8;;\
            with [1m-l[22m, print the author of each file

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-b\[1m-b, --escape[22m]8;;\
            print C-style escapes for nongraphic characters

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls--block-size\[1m--block-size=SIZE[22m]8;;\
            with [1m-l[22m, scale sizes by SIZE when printing them; e.g., '--block-size=M'; see SIZE format below

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-B\[1m-B, --ignore-backups[22m]8;;\
            do not list implied entries ending with ~

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-c\[1m-c[22m]8;;\     with [1m-lt[22m: sort by, and show, ctime (time of last change of file status information); with [1m-l[22m: show  ctime
            and sort by name; otherwise: sort by ctime, newest first

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-C\[1m-C[22m]8;;\     list entries by columns

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls--color\[1m--color[=WHEN][22m]8;;\
            color the output WHEN; more info below

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-d\[1m-d, --directory[22m]8;;\
            list directories themselves, not their contents

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-D\[1m-D, --dired[22m]8;;\
            generate output designed for Emacs' dired mode

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-f\[1m-f[22m]8;;\     same as [1m-a -U[0m

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-F\[1m-F, --classify[=WHEN][22m]8;;\
            append indicator (one of */=>@|) to entries WHEN

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls--file-type\[1m--file-type[22m]8;;\
            like [1m-F[22m, except do not append '*'

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls--format\[1m--format=WORD[22m]8;;\
            across,horizontal ([1m-x[22m), commas ([1m-m[22m), long ([1m-l[22m), single-column ([1m-1[22m), verbose ([1m-l[22m), vertical ([1m-C[22m)

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls--full-time\[1m--full-time[22m]8;;\
            like [1m-l --time-style[22m=[4mfull-iso[0m

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-g\[1m-g[22m]8;;\     like [1m-l[22m, but do not list owner

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls--group-directories-first\[1m--group-directories-first[22m]8;;\
            group directories before files

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-G\[1m-G, --no-group[22m]8;;\
            in a long listing, don't print group names

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-h\[1m-h, --human-readable[22m]8;;\
            with [1m-l [22mand [1m-s[22m, print sizes like 1K 234M 2G etc.

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls--si\[1m--si[22m]8;;\   likewise, but use powers of 1000 not 1024

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-H\[1m-H, --dereference-command-line[22m]8;;\
            follow symbolic links listed on the command line

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls--dereference-command-line-symlink-to-dir\[1m--dereference-command-line-symlink-to-dir[22m]8;;\
            follow each command line symbolic link that points to a directory

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls--hide\[1m--hide=PATTERN[22m]8;;\
            do not list implied entries matching shell PATTERN (overridden by [1m-a [22mor [1m-A[22m)

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls--hyperlink\[1m--hyperlink[=WHEN][22m]8;;\
            hyperlink file names WHEN

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls--indicator-style\[1m--indicator-style=WORD[22m]8;;\
            append  indicator  with  style  WORD to entry names: none (default), slash ([1m-p[22m), file-type ([1m--file-type[22m),
            classify ([1m-F[22m)

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-i\[1m-i, --inode[22m]8;;\
            print the index number of each file

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-I\[1m-I, --ignore=PATTERN[22m]8;;\
            do not list implied entries matching shell PATTERN

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-k\[1m-k, --kibibytes[22m]8;;\
            default to 1024-byte blocks for file system usage; used only with [1m-s [22mand per directory totals

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-l\[1m-l[22m]8;;\     use a long listing format

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-L\[1m-L, --dereference[22m]8;;\
            when showing file information for a symbolic link, show information for  the  file  the  link  references
            rather than for the link itself

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-m\[1m-m[22m]8;;\     fill width with a comma separated list of entries

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-n\[1m-n, --numeric-uid-gid[22m]8;;\
            like [1m-l[22m, but list numeric user and group IDs

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-N\[1m-N, --literal[22m]8;;\
            print entry names without quoting

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-o\[1m-o[22m]8;;\     like [1m-l[22m, but do not list group information

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-p\[1m-p, --indicator-style=slash[22m]8;;\
            append / indicator to directories

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-q\[1m-q, --hide-control-chars[22m]8;;\
            print ? instead of nongraphic characters

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls--show-control-chars\[1m--show-control-chars[22m]8;;\
            show nongraphic characters as-is; the default, unless program is 'ls' and output is a terminal

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-Q\[1m-Q, --quote-name[22m]8;;\
            enclose entry names in double quotes

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls--quoting-style\[1m--quoting-style=WORD[22m]8;;\
            use  quoting  style  WORD  for entry names: literal, locale, shell, shell-always, shell-escape, shell-es‐
            cape-always, c, escape (overrides QUOTING_STYLE environment variable)

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-r\[1m-r, --reverse[22m]8;;\
            reverse order while sorting

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-R\[1m-R, --recursive[22m]8;;\
            list subdirectories recursively

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-s\[1m-s, --size[22m]8;;\
            print the allocated size of each file, in blocks

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-S\[1m-S[22m]8;;\     sort by file size, largest first

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls--sort\[1m--sort=WORD[22m]8;;\
            change default 'name' sort to WORD: none ([1m-U[22m), size ([1m-S[22m), time ([1m-t[22m), version ([1m-v[22m), extension ([1m-X[22m),  name,
            width

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls--time\[1m--time=WORD[22m]8;;\
            select  which  timestamp  used  to display or sort; access time ([1m-u[22m): atime, access, use; metadata change
            time ([1m-c[22m): ctime, status; modified time (default): mtime, modification; birth time: birth, creation; with
            [1m-l[22m, WORD determines which time to show; with [1m--sort[22m=[4mtime[24m, sort by WORD (newest first)

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls--time-style\[1m--time-style=TIME_STYLE[22m]8;;\
            time/date format with [1m-l[22m; see TIME_STYLE below

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-t\[1m-t[22m]8;;\     sort by time, newest first; see [1m--time[0m

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-T\[1m-T, --tabsize=COLS[22m]8;;\
            assume tab stops at each COLS instead of 8

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-u\[1m-u[22m]8;;\     with [1m-lt[22m: sort by, and show, access time; with [1m-l[22m: show access time and sort by name; otherwise: sort  by
            access time, newest first

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-U\[1m-U[22m]8;;\     do not sort directory entries

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-v\[1m-v[22m]8;;\     natural sort of (version) numbers within text

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-w\[1m-w, --width=COLS[22m]8;;\
            set output width to COLS.  0 means no limit

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-x\[1m-x[22m]8;;\     list entries by lines instead of by columns

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-X\[1m-X[22m]8;;\     sort alphabetically by entry extension

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-Z\[1m-Z, --context[22m]8;;\
            print any security context of each file

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls--zero\[1m--zero[22m]8;;\
            end each output line with NUL, not newline

     ]8;;https://www.gnu.org/software/coreutils/manual/coreutils.html#ls-1\[1m-1[22m]8;;\     list one file per line

     ]8;;https://www.gnu.org/software/coreutils/ls#ls--help\[1m--help[22m]8;;\
            display this help and exit

     ]8;;https://www.gnu.org/software/coreutils/ls#ls--version\[1m--version[22m]8;;\
            output version information and exit

     The  SIZE  argument  is  an  integer and optional unit (example: 10K is 10*1024).  Units are K,M,G,T,P,E,Z,Y,R,Q
     (powers of 1024) or KB,MB,... (powers of 1000).  Binary prefixes can be used, too: KiB=K, MiB=M, and so on.

     The TIME_STYLE argument can be full-iso, long-iso, iso, locale, or  +FORMAT.   FORMAT  is  interpreted  like  in
     [1mdate[22m(1).   If  FORMAT is FORMAT1<newline>FORMAT2, then FORMAT1 applies to non-recent files and FORMAT2 to recent
     files.  TIME_STYLE prefixed with 'posix-' takes effect only outside the POSIX locale.  Also the TIME_STYLE envi‐
     ronment variable sets the default style to use.

     The WHEN argument defaults to 'always' and can also be 'auto' or 'never'.

     Using color to distinguish file types is disabled both by default and with [1m--color[22m=[4mnever[24m.  With [1m--color[22m=[4mauto[24m, ls
     emits color codes only when standard output is connected to a terminal.  The LS_COLORS environment variable  can
     change the settings.  Use the [1mdircolors[22m(1) command to set it.

   [1mExit status:[0m
     0      if OK,

     1      if minor problems (e.g., cannot access subdirectory),

     2      if serious trouble (e.g., cannot access command-line argument).

[1mAUTHOR[0m
     Written by Richard M. Stallman and David MacKenzie.

[1mREPORTING BUGS[0m
     Report bugs to: bug-coreutils@gnu.org
     GNU coreutils home page: <https://www.gnu.org/software/coreutils/>
     General help using GNU software: <https://www.gnu.org/gethelp/>
     Report any translation bugs to <https://translationproject.org/team/>

[1mCOPYRIGHT[0m
     Copyright © 2026 Free Software Foundation, Inc.  License GPLv3+: GNU GPL version 3 or later <https://gnu.org/li‐
     censes/gpl.html>.
     This  is free software: you are free to change and redistribute it.  There is NO WARRANTY, to the extent permit‐
     ted by law.

[1mSEE ALSO[0m
     [1mdircolors[22m(1)

     Full documentation <https://www.gnu.org/software/coreutils/ls>
     or available locally via: info '(coreutils) ls invocation'

GNU coreutils 9.11                                    April 2026                                                [4mLS[24m(1)
