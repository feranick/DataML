#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***************************************************
* Convert multiline text/file into a single line.
* By: Nicola Ferralis <feranick@hotmail.com>
* version v2025.04.05.1 
***************************************************
'''

import sys
import os.path
import argparse

__version__ = "2025.04.05.1" # Define version

def process_text(input_text):
    """Removes newline characters from a string."""
    return input_text.replace('\n', '')

def main():
    """Parses arguments and performs the conversion."""

    parser = argparse.ArgumentParser(
        description=__doc__, # Use module docstring for description
        formatter_class=argparse.RawDescriptionHelpFormatter # Preserve formatting
    )

    # Ensure only one action (text, file) is specified
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-t', '--text',
        metavar='\"<multi-line text>\"', # More descriptive metavar
        help='Directly provide the multiline text string to convert.'
    )
    group.add_argument(
        '-f', '--file',
        metavar='<input_file>',
        help='Specify an input file containing multiline text.'
    )

    parser.add_argument(
        '-v', '--version',
        action='version',
        version=f'%(prog)s {__version__}' # Show version information
    )
    
    # Add an optional output file argument
    parser.add_argument(
        '-o', '--output',
        metavar='<output_file>',
        help='(Optional) Specify a file to save the single-line output. '
             'If processing a file (-f) and -o is omitted, defaults to '
             '<input_file_root>_single.txt.'
    )

    args = parser.parse_args()

    single_line_data = ""
    output_destination_message = "to standard output" # Default message
    outfile = None # Initialize output file path

    try:
        if args.text:
            # --- Process direct text input ---
            print("Processing direct text input...")
            single_line_data = process_text(args.text)
            if args.output:
                 outfile = args.output # Use specified output file
                 output_destination_message = f"to file: {outfile}"

        elif args.file:
            # --- Process file input ---
            infile = args.file
            if not os.path.exists(infile):
                print(f"Error: Input file not found: {infile}", file=sys.stderr)
                return 1 # Return error code

            print(f"Opening input file: {infile}")
            
            # Determine default output file name if -o is not specified
            if not args.output:
                 file_root = os.path.splitext(infile)[0]
                 outfile = f"{file_root}_single.txt"
                 output_destination_message = f"to file: {outfile}"
            else:
                 outfile = args.output # Use specified output file
                 output_destination_message = f"to file: {outfile}"


            with open(infile, 'r', encoding='utf-8') as f: # Specify encoding
                # Read the whole file - acceptable for intended purpose
                # For very large files, consider line-by-line processing
                multi_line_data = f.read() 
                single_line_data = process_text(multi_line_data)

        # --- Output the result ---
        print("\nSingle line string:\n")
        print(single_line_data)

        # --- Save to file if required ---
        if outfile:
            print(f"\nSaving single line text {output_destination_message}")
            try:
                with open(outfile, "w", encoding='utf-8') as of: # Specify encoding
                    of.write(single_line_data)
                print("Save complete.")
            except IOError as e:
                print(f"Error: Could not write to output file {outfile}: {e}", file=sys.stderr)
                return 1 # Return error code

    except FileNotFoundError: # Catch potential issue during argument processing if needed
         print(f"Error: Input file specified not found.", file=sys.stderr)
         return 1
    except IOError as e:
        print(f"Error: An I/O error occurred: {e}", file=sys.stderr)
        return 1
    except Exception as e: # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return 1

    return 0 # Success

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
