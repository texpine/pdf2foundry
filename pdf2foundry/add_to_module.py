# class IncrementalModuleBuilder:
#     def add_content(self, existing_module_path: str, new_pdf: str):
#         """
#         Handles requirement #4 - avoiding duplicates
#         """
#         existing = self.load_module(existing_module_path)

#         # Hash all existing content
#         existing_hashes = {
#             self.hash_content(item) for item in existing.all_content
#         }

#         # Process new PDF
#         new_content = self.process_pdf(new_pdf)

#         # Only add genuinely new content
#         for item in new_content:
#             if self.hash_content(item) not in existing_hashes:
#                 existing.add_item(item)

#         return self.save_module(existing)
