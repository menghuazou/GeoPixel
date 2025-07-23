import torch


def interleav_wrap_chat(self, query, image, history = [], meta_instruction='', max_length=16384, hd_num=24):
    self.max_length = max_length
    prompt = ''
    if meta_instruction:
        prompt += f"""[UNUSED_TOKEN_146]system\n{meta_instruction}[UNUSED_TOKEN_145]\n"""
    for record in history:
        prompt += f"""[UNUSED_TOKEN_146]user\n{record[0]}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n{record[1]}[UNUSED_TOKEN_145]\n"""
    prompt += f"""[UNUSED_TOKEN_146]user\n{query}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"""

    image_nums = len(image)
    if image_nums == 1 and prompt.find('<ImageHere>') == -1: # 如果只有一张图，并且没有<ImageHere>，则在prompt前面添加<ImageHere>
        #print ('auto append image at the begining')
        prompt = '<ImageHere>' + prompt

    #从字符串的开始到第一个 <ImageHere> 标记之前的内容。由于在你的 prompt 中，<ImageHere> 是字符串的第一个元素，所以在它之前的文本为空（即没有内容），因此第一部分是空字符串 ''。
    parts = prompt.split('<ImageHere>') # 从<ImageHere>开始分割，分割成两个部分 ''和 'query'
    wrap_embeds, wrap_im_mask = [], []
    temp_len = 0
    need_bos = True

    if len(parts) != image_nums + 1:
        #raise ValueError('Invalid <ImageHere> prompt format.')
        print ('Waring! The image number != given position!')
    if image_nums > 1:
        hd_num = 6
    else:
        hu_num = hd_num
    for idx, part in enumerate(parts):
        if need_bos or len(part) > 0:
            part_tokens = self.tokenizer( # torch.Size([1, 1])， torch.Size([1, 33])
                part,
                return_tensors='pt',
                padding='longest',
                add_special_tokens=need_bos).to(self.device)
            if need_bos:
                need_bos = False

            part_embeds = self.model.tok_embeddings(
                part_tokens.input_ids)  # torch.Size([1, 1, 4096])， torch.Size([1, 33, 4096])
            wrap_embeds.append(part_embeds) # torch.Size([1, 1, 4096]) + torch.Size([1, 4081, 4096]) + torch.Size([1, 33, 4096])
            wrap_im_mask.append(torch.zeros(part_embeds.shape[:2])) # torch.Size([1, 1]) + torch.Size([1, 4081])+torch.Size([1, 33])
            temp_len += part_embeds.shape[1] # 1+4081+33=4115
        if idx < image_nums:
            img = self.encode_img(image[idx], hd_num) # torch.Size([1, 4081, 4096])
            wrap_embeds.append(img) #  text作为0，img作为1，append到wrap_embeds中
            wrap_im_mask.append(torch.ones(img.shape[:2])) #  torch.Size([1, 1]) + torch.Size([1, 4081])
            temp_len += img.shape[1] # 4082

        if temp_len > self.max_length:
            break
    #self.max_length：16384
    wrap_embeds = torch.cat(wrap_embeds, dim=1) # torch.Size([1, 4115, 4096])
    wrap_im_mask = torch.cat(wrap_im_mask, dim=1) # torch.Size([1, 4115])
    wrap_embeds = wrap_embeds[:, :self.max_length].to(self.device) # torch.Size([1, 4115, 4096])
    wrap_im_mask = wrap_im_mask[:, :self.max_length].to(self.device).bool() # torch.Size([1, 4115])
    inputs = {
        'inputs_embeds': wrap_embeds
    }
    return inputs, wrap_im_mask, temp_len   # text和image各自embeding，然后append到warp_embeds中。