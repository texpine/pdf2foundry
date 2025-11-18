# Proposed implementation structure
# class PDF2FoundryProcessor:
#     def __init__(self):
#         self.demo_pipeline = DemoPipeline()  # Fast, first 20 pages
#         self.full_pipeline = FullPipeline()  # Complete processing

#     async def process_demo(self, pdf_path: str) -> str:
#         """
#         Fast processing for first 20 pages
#         - Uses cached prompts
#         - Simplified extraction
#         - Lower quality tokens (SDXL Turbo)
#         - Returns in <2 minutes
#         """
#         pages = extract_pages(pdf_path, limit=20)
#         demo_result = await self.demo_pipeline.process(pages)
#         cache_key = self.cache_demo(demo_result)
#         return cache_key

#     async def process_full(self, pdf_path: str, cache_key: str) -> str:
#         """
#         Complete processing post-payment
#         - Can take 1-2 hours per 100 pages
#         - Full quality assets
#         - Complete extraction
#         - Builds on demo if available
#         """
#         demo_data = self.get_cached_demo(cache_key)
#         full_result = await self.full_pipeline.process(
#             pdf_path,
#             skip_pages=20 if demo_data else 0,
#             existing_data=demo_data
#         )
#         return self.package_module(full_result)
