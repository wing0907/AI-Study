import { useState } from 'react';
import { View, Text, TextInput, Pressable, Modal, StyleSheet, SafeAreaView } from 'react-native';

export default function Home() {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState('');

  return (
    <SafeAreaView style={s.root}>
      <View style={s.top}><Text style={s.brand}>LawAI</Text></View>
      <View style={s.card}><Text style={s.title}>지능형 리서치</Text></View>

      <View style={s.dock}>
        <Pressable style={s.plus} onPress={()=>setOpen(true)}><Text style={s.plusT}>＋</Text></Pressable>
        <TextInput value={text} onChangeText={setText} placeholder="무엇을 도와드릴까요?" style={s.input} />
        <Pressable style={s.send}><Text style={{color:'#fff'}}>전송</Text></Pressable>
      </View>

      <Modal transparent visible={open} animationType="fade">
        <Pressable style={s.backdrop} onPress={()=>setOpen(false)}>
          <View style={s.sheet}>
            <Pressable style={s.chip}><Text>🖼 이미지</Text></Pressable>
            <Pressable style={s.chip}><Text>🎙 음성</Text></Pressable>
            <Pressable style={s.chip}><Text>📎 파일</Text></Pressable>
          </View>
        </Pressable>
      </Modal>
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  root:{ flex:1, backgroundColor:'#0B2447' },
  top:{ padding:16, borderBottomWidth:1, borderColor:'#19376D' },
  brand:{ color:'#fff', fontWeight:'800', fontSize:18 },
  card:{ backgroundColor:'#fff', margin:16, padding:16, borderRadius:14, borderWidth:1, borderColor:'#19376D' },
  title:{ color:'#0f172a', fontWeight:'700' },
  dock:{ position:'absolute', left:0, right:0, bottom:20, flexDirection:'row', gap:8, paddingHorizontal:12 },
  plus:{ width:40, height:40, borderRadius:10, alignItems:'center', justifyContent:'center', borderWidth:1, borderColor:'#19376D', backgroundColor:'#fff' },
  plusT:{ color:'#0f172a', fontSize:18 },
  input:{ flex:1, height:40, backgroundColor:'#fff', borderRadius:10, paddingHorizontal:10, borderWidth:1, borderColor:'#19376D' },
  send:{ paddingHorizontal:14, borderRadius:10, backgroundColor:'#1D4ED8', height:40, alignItems:'center', justifyContent:'center' },
  backdrop:{ flex:1, backgroundColor:'rgba(0,0,0,0.4)', justifyContent:'flex-end' },
  sheet:{ flexDirection:'row', gap:8, backgroundColor:'#fff', margin:16, padding:10, borderRadius:12, borderWidth:1, borderColor:'#19376D' },
  chip:{ paddingVertical:6, paddingHorizontal:10, borderWidth:1, borderColor:'#19376D', borderRadius:999, backgroundColor:'#F6F8FC' },
});
